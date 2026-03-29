"""Load/save canonical processed close panels + metadata (elite brief §4.1)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

METADATA_NAME = "panel_metadata.json"
DEFAULT_PROCESSED_CLOSES_PARQUET = Path("data/processed/closes.parquet")
DEFAULT_PROCESSED_CLOSES_CSV = Path("data/processed/closes.csv")


def processed_paths(root: Path) -> tuple[Path, Path, Path]:
    proc = root / "data" / "processed"
    return proc / "closes.parquet", proc / "closes.csv", proc / METADATA_NAME


def load_processed_closes(root: Path) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load processed panel from parquet (preferred) or CSV under data/processed/."""
    pq, csv, meta_path = processed_paths(root)
    meta: dict[str, Any] | None = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
    else:
        return pd.DataFrame(), None
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).sort_values()
    df = df.sort_index()
    return df, meta


def save_processed_panel(
    closes: pd.DataFrame,
    root: Path,
    *,
    tickers_requested: list[str],
    source: str,
    fill_policy: str,
    missing_policy: str,
    first_valid_dates: dict[str, str],
    notes: str = "",
) -> None:
    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    out = closes.sort_index().copy()
    out.index.name = "date"
    pq, csv, meta_path = processed_paths(root)
    try:
        out.to_parquet(pq)
    except Exception:
        out.to_csv(csv)
    else:
        out.to_csv(csv)
    meta = {
        "source": source,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers_in_panel": list(out.columns),
        "tickers_requested": list(tickers_requested),
        "start_date": str(out.index[0].date()) if len(out) else None,
        "end_date": str(out.index[-1].date()) if len(out) else None,
        "n_rows": int(len(out)),
        "fill_policy": fill_policy,
        "missing_policy": missing_policy,
        "first_valid_date_by_ticker": first_valid_dates,
        "notes": notes,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
