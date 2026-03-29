"""Pre-backtest data quality report: missingness, staleness, jumps, coverage (elite brief §4.1)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnQuality:
    ticker: str
    first_valid_date: str | None
    last_valid_date: str | None
    missing_frac: float
    max_stale_run_bars: int
    jump_flag_count: int


@dataclass
class PanelQualityReport:
    n_rows: int
    n_cols: int
    date_start: str | None
    date_end: str | None
    rows_dropped_any_nan_frac: float
    overlap_bars_all_valid: int
    columns: list[ColumnQuality] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok_for_backtest: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["columns"] = [asdict(c) for c in self.columns]
        return d


def _stale_runs(mask: pd.Series) -> int:
    """Longest consecutive True run (missing/stale) in boolean mask."""
    if mask.empty:
        return 0
    m = mask.astype(int)
    # groups of consecutive 1s
    changes = m.diff().ne(0).cumsum()
    if not mask.any():
        return 0
    return int(m.groupby(changes).sum().max())


def _jump_count(series: pd.Series, z_threshold: float = 5.0) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 20:
        return 0
    lr = np.log(s / s.shift(1)).dropna()
    if len(lr) < 10:
        return 0
    mu, sig = float(lr.mean()), float(lr.std(ddof=1))
    if sig <= 0:
        return 0
    z = np.abs((lr - mu) / sig)
    return int((z > z_threshold).sum())


def compute_panel_quality(
    closes: pd.DataFrame,
    *,
    jump_z: float = 5.0,
    min_assets: int = 3,
    max_missing_frac_warn: float = 0.05,
) -> PanelQualityReport:
    """Analyze aligned close panel. Does not modify `closes`."""
    warnings: list[str] = []
    if closes.empty:
        return PanelQualityReport(
            n_rows=0,
            n_cols=0,
            date_start=None,
            date_end=None,
            rows_dropped_any_nan_frac=0.0,
            overlap_bars_all_valid=0,
            warnings=["empty panel"],
            ok_for_backtest=False,
        )

    idx = pd.DatetimeIndex(closes.index).sort_values()
    df = closes.reindex(idx).sort_index()
    tickers = list(df.columns)
    n_rows, n_cols = len(df), len(tickers)

    cols_out: list[ColumnQuality] = []
    for c in tickers:
        s = df[c]
        valid = s.notna()
        miss = float(1.0 - valid.mean()) if len(s) else 1.0
        first_i = s.first_valid_index()
        last_i = s.last_valid_index()
        first_s = str(first_i.date()) if first_i is not None and hasattr(first_i, "date") else None
        last_s = str(last_i.date()) if last_i is not None and hasattr(last_i, "date") else None
        stale_mask = ~valid
        max_stale = _stale_runs(stale_mask)
        jumps = _jump_count(s, jump_z)
        cols_out.append(
            ColumnQuality(
                ticker=str(c),
                first_valid_date=first_s,
                last_valid_date=last_s,
                missing_frac=miss,
                max_stale_run_bars=max_stale,
                jump_flag_count=jumps,
            )
        )
        if miss > max_missing_frac_warn:
            warnings.append(f"{c}: missing_frac={miss:.3f} exceeds {max_missing_frac_warn}")
        if jumps > 0:
            warnings.append(f"{c}: {jumps} log-return jumps beyond {jump_z} sigma")

    complete = df.dropna(how="any")
    overlap = len(complete)
    dropped_frac = 1.0 - (overlap / n_rows) if n_rows else 0.0

    if overlap < min_assets * 50:
        warnings.append(
            f"overlap_bars_all_valid={overlap} may be too short for stable backtest (warmup + tests)"
        )

    date_start = str(df.index[0].date()) if n_rows else None
    date_end = str(df.index[-1].date()) if n_rows else None

    ok = overlap >= min_assets * 50 and n_cols >= min_assets and not df.empty

    return PanelQualityReport(
        n_rows=n_rows,
        n_cols=n_cols,
        date_start=date_start,
        date_end=date_end,
        rows_dropped_any_nan_frac=float(dropped_frac),
        overlap_bars_all_valid=int(overlap),
        columns=cols_out,
        warnings=warnings,
        ok_for_backtest=ok,
    )


def assert_panel_ok(report: PanelQualityReport, *, strict: bool = True) -> None:
    if not report.ok_for_backtest and strict:
        msg = "; ".join(report.warnings) or "panel failed quality checks"
        raise ValueError(f"Data quality gate: {msg}")


def write_quality_report(report: PanelQualityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
