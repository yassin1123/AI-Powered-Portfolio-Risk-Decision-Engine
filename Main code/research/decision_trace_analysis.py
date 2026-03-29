"""Primary decision driver mix from decision_log (decision_priority)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_decision_breakdown(decision_log_path: Path, out_path: Path) -> None:
    df = pd.read_csv(decision_log_path)
    if "decision_priority" not in df.columns:
        raise ValueError("decision_log must contain decision_priority")
    vc = df["decision_priority"].value_counts(dropna=False)
    total = int(vc.sum())
    rows = []
    for name, count in vc.items():
        rows.append(
            {
                "decision_priority": name,
                "count": int(count),
                "pct": round(100.0 * float(count) / total, 4) if total else 0.0,
            }
        )
    out = pd.DataFrame(rows).sort_values("count", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Decision breakdown CSV")
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
        default=Path("research/outputs/decision_breakdown.csv"),
    )
    args = ap.parse_args()
    write_decision_breakdown(args.decision_log, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
