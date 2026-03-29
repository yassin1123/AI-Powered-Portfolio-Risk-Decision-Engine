"""Drawdown, corr_z, and regime overlay (matplotlib)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _regime_numeric(s: pd.Series) -> np.ndarray:
    mapping = {"CALM": 0.0, "TRANSITION": 1.0, "STRESSED": 2.0}
    return s.map(lambda x: mapping.get(str(x).upper(), 0.5)).astype(float).values


def plot_killer_overlay(equity_path: Path, decision_log_path: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    raw = pd.read_csv(equity_path)
    dts = pd.to_datetime(raw.iloc[:, 0])
    vals = raw.iloc[:, 1].astype(float).values
    run_dd = vals / np.maximum.accumulate(vals) - 1.0

    dl = pd.read_csv(decision_log_path)
    n = min(len(vals), len(dl))
    if n < 2:
        raise ValueError("Need at least 2 aligned rows for overlay plot")
    dts = dts.iloc[:n]
    run_dd = run_dd[:n]
    corr_z = dl["corr_z"].astype(float).values[:n]
    regime_y = _regime_numeric(dl["regime"].iloc[:n])

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].fill_between(dts, run_dd * 100, 0, color="steelblue", alpha=0.35)
    axes[0].plot(dts, run_dd * 100, color="navy", lw=0.8)
    axes[0].set_ylabel("Drawdown %")
    axes[0].set_title("Portfolio drawdown")

    axes[1].plot(dts, corr_z, color="darkorange", lw=0.9)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel("corr_z")
    axes[1].set_title("Correlation z-score")

    axes[2].step(dts, regime_y, where="post", color="seagreen", lw=1.0)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["CALM", "TRANSITION", "STRESSED"])
    axes[2].set_ylabel("Regime")
    axes[2].set_title("Regime (step)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha="right")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Killer overlay chart")
    ap.add_argument(
        "equity",
        type=Path,
        nargs="?",
        default=Path("research/outputs/equity_curve.csv"),
    )
    ap.add_argument(
        "decision_log",
        type=Path,
        nargs="?",
        default=Path("research/outputs/decision_log.csv"),
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("research/figures/killer_overlay.png"),
    )
    args = ap.parse_args()
    plot_killer_overlay(args.equity, args.decision_log, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
