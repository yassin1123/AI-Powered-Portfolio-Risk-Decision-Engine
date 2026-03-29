"""Compare stress scenario portfolio PnL to worst realized window (placeholder hook)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--equity", type=Path, default=Path("research/outputs/equity_curve.csv"))
    p.add_argument("--stress-pnl", type=float, default=-0.12, help="Example scenario portfolio PnL")
    args = p.parse_args()
    if not args.equity.exists():
        print("No equity curve; export from backtest extension or use ladder run.")
        return
    eq = pd.read_csv(args.equity, index_col=0).iloc[:, 0]
    dd = eq / eq.cummax() - 1
    print("Worst realized drawdown:", float(dd.min()))
    print("Example stress scenario PnL:", args.stress_pnl)


if __name__ == "__main__":
    main()
