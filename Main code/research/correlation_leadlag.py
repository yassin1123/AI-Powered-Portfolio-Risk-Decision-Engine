"""Forward outcomes conditional on corr_z (same path as decision_log)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

Z_THRESHOLD = 1.5
VAR_DAY_BREACH = 0.02  # simple realized tail proxy for forward window


def _forward_stats(pnl: np.ndarray, start: int, horizon: int) -> tuple[float, float, int]:
    """Return (max_drawdown_frac, ann_vol_proxy, var_breach_flag) over next `horizon` bars."""
    end = min(start + 1 + horizon, len(pnl))
    chunk = pnl[start + 1 : end]
    if len(chunk) == 0:
        return 0.0, 0.0, 0
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for r in chunk:
        eq *= float(np.exp(r))
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
    vol = float(np.std(chunk) * np.sqrt(252)) if len(chunk) > 1 else 0.0
    breach = int(np.any(chunk < -VAR_DAY_BREACH))
    return float(mdd), vol, breach


def compute_leadlag_table(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    pnl = df["pnl_frac"].astype(float).values
    z = df["corr_z"].astype(float).values
    n = len(df)
    rows_out: list[dict[str, float | int | str]] = []

    def collect(mask: np.ndarray, label: str) -> None:
        mdds: list[float] = []
        vols: list[float] = []
        breaches = 0
        k = 0
        for i in range(n):
            if not mask[i]:
                continue
            if i + horizon >= len(pnl):
                continue
            mdd, vol, br = _forward_stats(pnl, i, horizon)
            mdds.append(mdd)
            vols.append(vol)
            breaches += br
            k += 1
        if k == 0:
            rows_out.append(
                {
                    "condition": label,
                    "horizon_bars": horizon,
                    "n": 0,
                    "avg_fwd_dd": np.nan,
                    "avg_fwd_vol_ann": np.nan,
                    "var_breach_rate_fwd": np.nan,
                }
            )
            return
        rows_out.append(
            {
                "condition": label,
                "horizon_bars": horizon,
                "n": k,
                "avg_fwd_dd": float(np.mean(mdds)),
                "avg_fwd_vol_ann": float(np.mean(vols)),
                "var_breach_rate_fwd": breaches / k,
            }
        )

    high = z > Z_THRESHOLD
    base = np.abs(z) <= Z_THRESHOLD
    collect(high, f"corr_z > {Z_THRESHOLD}")
    collect(base, f"|corr_z| <= {Z_THRESHOLD} (baseline)")
    return pd.DataFrame(rows_out)


def write_leadlag_summary(decision_log_path: Path, out_path: Path) -> None:
    df = pd.read_csv(decision_log_path)
    if "pnl_frac" not in df.columns or "corr_z" not in df.columns:
        raise ValueError("decision_log must contain pnl_frac and corr_z")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tbl = compute_leadlag_table(df, horizon=5)
    tbl.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Lead-lag summary from decision_log")
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
        default=Path("research/outputs/leadlag_summary.csv"),
    )
    args = ap.parse_args()
    write_leadlag_summary(args.decision_log, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
