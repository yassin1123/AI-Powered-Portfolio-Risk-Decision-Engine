"""Stats around regime label changes from backtest decision log."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_transitions(log_path: Path) -> None:
    if not log_path.exists():
        print("No log; run backtest and export decision_log.csv first.")
        return
    df = pd.read_csv(log_path)
    if "regime" not in df.columns or "corr_z" not in df.columns:
        print("Missing columns")
        return
    chg = df["regime"] != df["regime"].shift(1)
    idx = df.index[chg.fillna(False)].tolist()
    before_corr_z = []
    after_corr_z = []
    for i in idx:
        if i > 0 and i < len(df) - 1:
            before_corr_z.append(float(df["corr_z"].iloc[i - 1]))
            after_corr_z.append(float(df["corr_z"].iloc[i]))
    if before_corr_z:
        print(
            "Mean |corr_z| change at transition:",
            float(np.mean(np.abs(np.array(after_corr_z) - np.array(before_corr_z)))),
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=Path("research/outputs/decision_log.csv"))
    args = p.parse_args()
    summarize_transitions(args.log)


if __name__ == "__main__":
    main()
