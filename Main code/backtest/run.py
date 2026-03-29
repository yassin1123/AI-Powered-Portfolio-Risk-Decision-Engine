"""CLI: synthetic backtest, placebo, five-strategy ladder, lead-lag, plots, key findings."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from pre.settings import load_settings


def synthetic_closes(
    n: int = 400,
    k: int = 6,
    seed: int = 42,
    *,
    daily_drift: float = 0.00028,
    vol_per_bar: float = 0.01,
) -> pd.DataFrame:
    """Correlated log-normal-ish paths. **Drift** gives a positive equity risk premium so the default
    ladder is not dominated by (near) zero-mean noise + risk-free hurdle alone."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    shock = rng.standard_normal((n, k)) * vol_per_bar + daily_drift
    x = shock.cumsum(axis=0)
    px = 100 * np.exp(x)
    cols = [f"AS{i}" for i in range(k)]
    return pd.DataFrame(px, index=dates, columns=cols)


def _row(strategy: str, m: dict[str, float]) -> dict[str, float | str]:
    return {
        "Strategy": strategy,
        "Sharpe": round(float(m.get("sharpe", 0)), 4),
        "Sharpe_gross": round(float(m.get("sharpe_gross", m.get("sharpe", 0))), 4),
        "max_dd": round(float(m.get("max_dd", 0)), 6),
        "max_dd_gross": round(float(m.get("max_dd_gross", m.get("max_dd", 0))), 6),
        "var_breach_rate": float(m.get("var_breach_rate", 0)),
        "var_breach_cluster_max": float(m.get("var_breach_cluster_max", 0)),
        "mean_turnover": round(float(m.get("mean_turnover", 0)), 6),
        "cagr": round(float(m.get("cagr", 0)), 6),
        "cagr_gross": round(float(m.get("cagr_gross", m.get("cagr", 0))), 6),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Portfolio risk engine backtest")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic price panel (stress/demo; headline path is real data)",
    )
    p.add_argument("--panel", type=Path, default=None, help="Explicit path to processed closes CSV/parquet")
    p.add_argument("--placebo", action="store_true", help="Only run random-signal placebo")
    p.add_argument("--no-extras", action="store_true", help="Skip lead-lag, breakdown, plots, key_findings")
    p.add_argument(
        "--leadlag",
        action="store_true",
        help="Only run lead-lag script on existing research/outputs/decision_log.csv",
    )
    p.add_argument("--no-qc", action="store_true", help="Skip pre-backtest data quality report")
    p.add_argument(
        "--qc-out",
        type=Path,
        default=None,
        help="Write data_quality_report.json (default: research/outputs/data_quality_report.json)",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--cost-sweep",
        action="store_true",
        help="Write cost_sensitivity.csv for cost_bps in {5,10,20} (full mode only) and exit",
    )
    p.add_argument(
        "--rebalance-every",
        type=int,
        default=None,
        help="Override backtest.rebalance_every_bars for this run",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = args.config or (root / "config.yaml")
    settings = load_settings(cfg)
    out_dir = root / "research" / "outputs"
    fig_dir = root / "research" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.leadlag:
        from research.correlation_leadlag import write_leadlag_summary

        write_leadlag_summary(out_dir / "decision_log.csv", out_dir / "leadlag_summary.csv")
        print(f"Wrote {out_dir / 'leadlag_summary.csv'}")
        return

    if args.synthetic:
        closes = synthetic_closes(
            daily_drift=settings.backtest.synthetic_daily_drift,
            vol_per_bar=settings.backtest.synthetic_vol_per_bar,
        )
    elif args.panel is not None:
        pth = args.panel
        if pth.suffix.lower() in (".parquet", ".pq"):
            closes = pd.read_parquet(pth)
        else:
            closes = pd.read_csv(pth, index_col=0, parse_dates=True)
        closes.index = pd.DatetimeIndex(pd.to_datetime(closes.index)).sort_values()
    else:
        from data.panel_store import load_processed_closes

        closes, _meta = load_processed_closes(root)
        if closes.empty:
            print(
                "No processed panel found. Build real data:\n"
                "  python scripts/build_data_panel.py\n"
                "Or pass --panel PATH / use --synthetic for a demo panel.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not args.no_qc:
        from data.data_quality import compute_panel_quality, write_quality_report

        qc = compute_panel_quality(closes)
        qc_path = args.qc_out or (out_dir / "data_quality_report.json")
        write_quality_report(qc, qc_path)
        print(f"Wrote {qc_path} (ok_for_backtest={qc.ok_for_backtest})")
        if not qc.ok_for_backtest:
            print("Quality warnings:\n  " + "\n  ".join(qc.warnings), file=sys.stderr)
            sys.exit(1)

    if args.cost_sweep:
        sweep_rows: list[dict[str, float | str]] = []
        for bps in (5.0, 10.0, 20.0):
            pc = settings.portfolio.model_copy(update={"cost_bps": bps})
            s2 = settings.model_copy(update={"portfolio": pc})
            r = run_backtest(
                closes,
                s2,
                mode="full",
                rebalance_every=args.rebalance_every,
            )
            sweep_rows.append(
                {
                    "cost_bps": bps,
                    "cagr": round(float(r.metrics.get("cagr", 0)), 6),
                    "cagr_gross": round(float(r.metrics.get("cagr_gross", 0)), 6),
                    "sharpe": round(float(r.metrics.get("sharpe", 0)), 4),
                    "max_dd": round(float(r.metrics.get("max_dd", 0)), 6),
                    "mean_turnover": round(float(r.metrics.get("mean_turnover", 0)), 6),
                }
            )
        cpath = out_dir / "cost_sensitivity.csv"
        with open(cpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"Wrote {cpath}")
        return

    rows: list[dict[str, float | str]] = []
    export_res = None
    if args.placebo:
        export_res = run_backtest(
            closes, settings, mode="placebo_random", rebalance_every=args.rebalance_every
        )
        rows.append(_row("placebo_random_signals", export_res.metrics))
    else:
        ladder = [
            ("Baseline EW", "baseline"),
            ("Inverse vol (RP proxy)", "vol_target_only"),
            ("Momentum 50% long", "momentum_naive"),
            ("Signals only", "signals_only"),
            ("Correlation signal only", "corr_signal_only"),
            ("Full system", "full"),
        ]
        rb = args.rebalance_every
        for label, mode in ladder:
            res = run_backtest(closes, settings, mode=mode, rebalance_every=rb)  # type: ignore[arg-type]
            rows.append(_row(label, res.metrics))
            if mode == "full":
                export_res = res
        res_p = run_backtest(closes, settings, mode="placebo_random", rebalance_every=rb)
        rows.append(_row("placebo_random_signals", res_p.metrics))
        if export_res is None:
            export_res = run_backtest(closes, settings, mode="full", rebalance_every=rb)

    ladder_path = out_dir / "ladder_table.csv"
    fieldnames = list(rows[0].keys())
    with open(ladder_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if export_res is not None:
        pd.DataFrame(export_res.decision_log).to_csv(out_dir / "decision_log.csv", index=False)
        export_res.equity.rename("equity_net").to_csv(out_dir / "equity_curve.csv")
        if export_res.equity_gross is not None:
            export_res.equity_gross.rename("equity_gross").to_csv(
                out_dir / "equity_curve_gross.csv", index=True
            )

    run_extras = not args.no_extras and not args.placebo
    if run_extras and export_res is not None:
        from research.correlation_leadlag import write_leadlag_summary

        write_leadlag_summary(out_dir / "decision_log.csv", out_dir / "leadlag_summary.csv")

        from research.decision_trace_analysis import write_decision_breakdown

        write_decision_breakdown(out_dir / "decision_log.csv", out_dir / "decision_breakdown.csv")

        try:
            from research.plot_killer_overlay import plot_killer_overlay

            plot_killer_overlay(
                out_dir / "equity_curve.csv",
                out_dir / "decision_log.csv",
                fig_dir / "killer_overlay.png",
            )
        except ImportError:
            print("matplotlib not installed; skip killer_overlay.png", file=sys.stderr)

        from scripts.update_key_findings import patch_key_findings

        patch_key_findings(
            root / "research" / "key_findings.md",
            ladder_path,
            out_dir / "leadlag_summary.csv",
            out_dir / "decision_breakdown.csv",
        )

    print(f"Wrote {ladder_path} and decision artifacts")


if __name__ == "__main__":
    main()
