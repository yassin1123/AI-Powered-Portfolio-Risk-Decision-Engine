"""Ablation runner → research/outputs/ablation_results.csv (elite brief §5.2)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.ablation import AblationFlags
from backtest.engine import run_backtest
from backtest.run import synthetic_closes
from data.panel_store import load_processed_closes
from pre.settings import load_settings


def main() -> None:
    p = argparse.ArgumentParser(description="Run ablation grid on one price panel")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--panel", type=Path, default=None)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    settings = load_settings(args.config or (ROOT / "config.yaml"))
    if args.synthetic:
        closes = synthetic_closes(
            daily_drift=settings.backtest.synthetic_daily_drift,
            vol_per_bar=settings.backtest.synthetic_vol_per_bar,
            n=800,
        )
    elif args.panel:
        import pandas as pd

        closes = pd.read_csv(args.panel, index_col=0, parse_dates=True)
        closes.index = pd.DatetimeIndex(pd.to_datetime(closes.index))
    else:
        closes, _ = load_processed_closes(ROOT)
        if closes.empty:
            print("No panel: use --synthetic or build_data_panel", file=sys.stderr)
            sys.exit(1)

    grid: list[tuple[str, AblationFlags]] = [
        ("full_baseline", AblationFlags()),
        ("no_correlation_signal", AblationFlags(use_correlation_signal=False)),
        ("no_regime_gating", AblationFlags(use_regime_gating=False)),
        ("no_anomaly_gating", AblationFlags(use_anomaly_gating=False)),
        ("no_vol_target", AblationFlags(use_vol_target=False)),
        ("no_transaction_costs", AblationFlags(use_transaction_costs=False)),
        ("no_decision_engine", AblationFlags(use_decision_engine=False)),
        ("no_turnover_cap", AblationFlags(use_turnover_cap=False)),
    ]

    out_dir = ROOT / "research" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_results.csv"
    fieldnames = [
        "ablation",
        "cagr",
        "cagr_gross",
        "sharpe",
        "sharpe_gross",
        "max_dd",
        "max_dd_gross",
        "mean_turnover",
        "var_breach_rate",
    ]
    rows: list[dict[str, float | str]] = []
    for name, flags in grid:
        res = run_backtest(closes, settings, mode="full", ablation=flags)
        m = res.metrics
        rows.append(
            {
                "ablation": name,
                "cagr": round(float(m.get("cagr", 0)), 6),
                "cagr_gross": round(float(m.get("cagr_gross", 0)), 6),
                "sharpe": round(float(m.get("sharpe", 0)), 4),
                "sharpe_gross": round(float(m.get("sharpe_gross", 0)), 4),
                "max_dd": round(float(m.get("max_dd", 0)), 6),
                "max_dd_gross": round(float(m.get("max_dd_gross", 0)), 6),
                "mean_turnover": round(float(m.get("mean_turnover", 0)), 6),
                "var_breach_rate": float(m.get("var_breach_rate", 0)),
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
