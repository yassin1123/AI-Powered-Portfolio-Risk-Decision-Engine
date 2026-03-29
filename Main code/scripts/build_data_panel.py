"""Canonical data build: raw OHLCV per ticker, aligned processed closes, metadata (elite brief §4.1).

Run from `Main code`:
  python scripts/build_data_panel.py [--profile core] [--start 2010-01-01]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.panel_store import save_processed_panel
from data.universe import get_tickers


def _download_ohlcv(ticker: str, start: str) -> tuple[str, pd.DataFrame | None]:
    try:
        tk = yf.Ticker(ticker)
        h = tk.history(start=start, auto_adjust=True, actions=False)
    except Exception:
        return ticker, None
    if h is None or h.empty:
        return ticker, None
    return ticker, h


def main() -> None:
    p = argparse.ArgumentParser(description="Build raw + processed price panel from yfinance")
    p.add_argument("--profile", default="core", help="universe profile: core | full")
    p.add_argument("--start", default="2010-01-01", help="History start date (ISO)")
    p.add_argument("--config-root", type=Path, default=None, help="Project root (default: Main code)")
    args = p.parse_args()
    root = args.config_root or ROOT

    tickers = get_tickers(args.profile)
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    max_workers = min(8, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_download_ohlcv, t, args.start): t for t in tickers}
        for fut in as_completed(futs):
            t, h = fut.result()
            if h is not None and not h.empty:
                h.to_csv(raw_dir / f"{t.replace('^', '_')}.csv")
                frames[t] = h

    if len(frames) < 3:
        print("Too few tickers downloaded; check network and symbols.", file=sys.stderr)
        sys.exit(1)

    close_cols: dict[str, pd.Series] = {}
    first_dates: dict[str, str] = {}
    for t, h in frames.items():
        if "Close" not in h.columns:
            continue
        s = pd.to_numeric(h["Close"], errors="coerce")
        s = s.rename(t)
        close_cols[t] = s
        fv = s.first_valid_index()
        if fv is not None:
            first_dates[t] = str(fv.date()) if hasattr(fv, "date") else str(fv)

    closes = pd.DataFrame(close_cols).sort_index()
    closes = closes.ffill(limit=3).bfill(limit=3)

    # Common history: no silent fantasy before first real print (brief §4.1)
    valid_starts = []
    for c in closes.columns:
        fv = closes[c].first_valid_index()
        if fv is not None:
            valid_starts.append(fv)
    if not valid_starts:
        print("No valid close history.", file=sys.stderr)
        sys.exit(1)
    start_common = max(valid_starts)
    closes = closes.loc[start_common:].copy()
    closes = closes.dropna(how="any")

    notes = (
        f"Universe profile={args.profile}; aligned inner-join after ffill/bfill limit 3; "
        f"trimmed from max(first_valid)={start_common}."
    )
    save_processed_panel(
        closes,
        root,
        tickers_requested=tickers,
        source="yfinance",
        fill_policy="ffill_limit_3_bfill_limit_3",
        missing_policy="drop_row_if_any_nan_after_align",
        first_valid_dates=first_dates,
        notes=notes,
    )
    meta_path = root / "data" / "processed" / "panel_metadata.json"
    print(f"Wrote {len(closes)} rows x {len(closes.columns)} cols → data/processed/")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
