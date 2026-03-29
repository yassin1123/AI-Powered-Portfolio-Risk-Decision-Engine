"""Walk-forward: fit correlation z-thresholds on past-only train, OOS metrics on test blocks (elite brief §5.1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from backtest.corr_z_path import correlation_z_series
from backtest.engine import BacktestResult, run_backtest
from backtest.evaluation import summarize_backtest
from pre.settings import AppSettings, load_settings

Mode = Literal[
    "full",
    "baseline",
    "vol_target_only",
    "signals_only",
    "corr_signal_only",
    "placebo_random",
]


@dataclass
class WalkforwardSlice:
    window_id: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_end_date: str
    test_start_date: str
    test_end_date: str
    fitted_z_high: float
    fitted_z_low: float
    result: BacktestResult
    oos_metrics: dict[str, float]


def walk_forward(
    closes: pd.DataFrame,
    settings: AppSettings,
    *,
    mode: Mode = "full",
    warmup: int = 130,
) -> tuple[list[WalkforwardSlice], list[dict[str, Any]]]:
    tr = settings.backtest.walkforward_train_bars
    te = settings.backtest.walkforward_test_bars
    z_series = correlation_z_series(closes, settings, warmup=warmup)
    z_series = z_series.reindex(closes.index)

    slices: list[WalkforwardSlice] = []
    manifest_rows: list[dict[str, Any]] = []

    i = tr
    wid = 0
    while i + te < len(closes):
        z_train = z_series.iloc[:i].dropna()
        if len(z_train) < 40:
            i += te
            continue
        zh = float(np.quantile(z_train.values, 0.9))
        zl = float(np.quantile(z_train.values, 0.1))
        new_corr = settings.correlation_signal.model_copy(update={"z_high": zh, "z_low": zl})
        wf_settings = settings.model_copy(update={"correlation_signal": new_corr})

        sub = closes.iloc[: i + te].copy()
        res = run_backtest(sub, wf_settings, mode=mode, warmup=warmup)

        oos_eq = res.equity.iloc[-min(te, len(res.equity)) :]
        oos_turn = res.turnover.reindex(oos_eq.index).fillna(0.0)
        oos_m = summarize_backtest(oos_eq, oos_turn, settings.risk_free_annual)
        oos_m["var_breach_rate"] = float(res.metrics.get("var_breach_rate", 0.0))

        t_start = i
        t_end = min(i + te, len(closes)) - 1
        row = {
            "window_id": wid,
            "train_end_idx": i,
            "test_start_idx": t_start,
            "test_end_idx": t_end,
            "train_end_date": str(closes.index[i - 1].date()) if i > 0 else "",
            "test_start_date": str(closes.index[t_start].date()),
            "test_end_date": str(closes.index[t_end].date()),
            "fitted_z_high": zh,
            "fitted_z_low": zl,
        }
        manifest_rows.append({**row, **{f"oos_{k}": v for k, v in oos_m.items()}})

        slices.append(
            WalkforwardSlice(
                window_id=wid,
                train_end_idx=i,
                test_start_idx=t_start,
                test_end_idx=t_end,
                train_end_date=row["train_end_date"],
                test_start_date=row["test_start_date"],
                test_end_date=row["test_end_date"],
                fitted_z_high=zh,
                fitted_z_low=zl,
                result=res,
                oos_metrics=oos_m,
            )
        )
        wid += 1
        i += te

    return slices, manifest_rows


def write_walkforward_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _cli() -> None:
    import argparse

    from data.panel_store import load_processed_closes

    p = argparse.ArgumentParser(description="Walk-forward OOS evaluation (frozen corr thresholds per fold)")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--panel", type=Path, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--mode", default="full")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Lower backtest MC VaR paths for a quicker smoke run (not for final research)",
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = load_settings(args.config or (root / "config.yaml"))
    if args.fast:
        settings = settings.model_copy(
            update={
                "backtest": settings.backtest.model_copy(
                    update={"var_mc_sims": min(250, settings.backtest.var_mc_sims)}
                )
            }
        )
    if args.synthetic:
        from backtest.run import synthetic_closes

        n_syn = 650 if args.fast else 900
        closes = synthetic_closes(
            daily_drift=settings.backtest.synthetic_daily_drift,
            vol_per_bar=settings.backtest.synthetic_vol_per_bar,
            n=n_syn,
        )
    elif args.panel:
        closes = pd.read_csv(args.panel, index_col=0, parse_dates=True)
        closes.index = pd.DatetimeIndex(pd.to_datetime(closes.index))
    else:
        closes, _ = load_processed_closes(root)
        if closes.empty:
            raise SystemExit("No processed panel: run scripts/build_data_panel.py or --synthetic")

    slices, manifest = walk_forward(closes, settings, mode=args.mode)  # type: ignore[arg-type]
    out = args.out or (root / "research" / "outputs" / "walkforward_manifest.json")
    write_walkforward_manifest(manifest, out)
    print(f"Wrote {out} ({len(slices)} windows)")


if __name__ == "__main__":
    _cli()
