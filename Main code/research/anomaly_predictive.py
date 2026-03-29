"""Lag anomaly intensity vs future vol / drawdown (run on saved decision logs or panel)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def simple_predictive_table(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(log_path)
    if "anomaly_count" not in df.columns or "pnl_frac" not in df.columns:
        return pd.DataFrame()
    hi = df["anomaly_count"] >= 3
    fut_vol = df["pnl_frac"].rolling(5).std().shift(-5)
    return pd.DataFrame(
        {
            "high_anomaly_future_5d_vol": [float(fut_vol[hi].mean())],
            "low_anomaly_future_5d_vol": [float(fut_vol[~hi].mean())],
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=Path("research/outputs/decision_log.csv"))
    args = p.parse_args()
    t = simple_predictive_table(args.log)
    print(t.to_string())


if __name__ == "__main__":
    main()
