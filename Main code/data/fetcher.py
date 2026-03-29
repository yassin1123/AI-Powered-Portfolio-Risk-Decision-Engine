"""Data ingestion: yfinance batch, validation, circular buffer, async + simulation replay."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from pre.settings import AppSettings


def _extract_close_panel(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            closes = raw["Close"].copy()
        elif "Adj Close" in level0:
            closes = raw["Adj Close"].copy()
        else:
            closes = raw.xs("Close", axis=1, level=0, drop_level=True)
    else:
        if "Close" in raw.columns:
            closes = raw[["Close"]].copy()
            if len(tickers) == 1:
                closes.columns = tickers
        else:
            closes = raw.copy()
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers[0])
    closes = closes.reindex(columns=tickers)
    return closes


def _download_one_close(ticker: str, period: str) -> tuple[str, pd.Series | None]:
    """Single-ticker history — avoids yfinance batch NoneType / partial-fail bugs."""
    try:
        tk = yf.Ticker(ticker)
        h = tk.history(period=period, auto_adjust=True, actions=False)
    except (TypeError, AttributeError, KeyError, ValueError):
        return ticker, None
    except Exception:
        return ticker, None
    if h is None or h.empty or "Close" not in h.columns:
        return ticker, None
    s = pd.to_numeric(h["Close"], errors="coerce")
    if s.notna().sum() < 10:
        return ticker, None
    s = s.rename(ticker)
    return ticker, s


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Schema validation, 5-sigma outlier flagging on log-returns."""
    issues: dict[str, Any] = {"ticker": ticker, "outliers": []}
    if df.empty:
        issues["error"] = "empty"
        return df, issues
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        issues["error"] = f"missing_columns:{missing}"
        return df, issues
    close = df["Close"].astype(float)
    lr = np.log(close / close.shift(1)).dropna()
    if len(lr) > 10:
        mu, sig = float(lr.mean()), float(lr.std(ddof=1))
        if sig > 0:
            z = np.abs((lr - mu) / sig)
            bad = z[z > 5.0]
            if len(bad) > 0:
                issues["outliers"] = bad.index.astype(str).tolist()[:20]
    return df, issues


@dataclass
class CircularBuffer:
    """Rolling window of adjusted closes: O(1) append via pointer (brief §8.1)."""

    max_rows: int
    tickers: list[str]
    closes: np.ndarray = field(init=False)
    dates: np.ndarray = field(init=False)
    write_idx: int = field(default=0, init=False)
    filled: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        n = len(self.tickers)
        self.closes = np.full((self.max_rows, n), np.nan, dtype=np.float64)
        self.dates = np.empty(self.max_rows, dtype="datetime64[ns]")

    def load_initial(self, price_matrix: np.ndarray, date_index: pd.DatetimeIndex) -> None:
        """price_matrix shape (T, n_assets), last max_rows kept."""
        n = min(price_matrix.shape[0], self.max_rows)
        sl = price_matrix[-n:]
        self.closes[:n] = sl
        di = date_index[-n:].values.astype("datetime64[ns]")
        self.dates[:n] = di
        self.filled = n
        self.write_idx = n % self.max_rows

    def append_row(self, dt: datetime, prices: np.ndarray) -> None:
        """Single new row (1, n_assets)."""
        i = self.write_idx
        self.dates[i] = np.datetime64(dt)
        self.closes[i] = prices
        self.write_idx = (i + 1) % self.max_rows
        self.filled = min(self.filled + 1, self.max_rows)

    def get_closes_df(self) -> pd.DataFrame:
        """Return DataFrame in chronological order (oldest first)."""
        if self.filled < self.max_rows:
            idx = np.arange(self.filled)
            data = self.closes[: self.filled]
            dts = self.dates[: self.filled]
        else:
            idx = np.concatenate([np.arange(self.write_idx, self.max_rows), np.arange(0, self.write_idx)])
            data = self.closes[idx]
            dts = self.dates[idx]
        return pd.DataFrame(data, index=pd.DatetimeIndex(dts), columns=self.tickers)


class DataFetcher:
    def __init__(self, settings: AppSettings, tickers: list[str]) -> None:
        self.settings = settings
        self.tickers = list(tickers)
        self.buffer = CircularBuffer(settings.lookback_days, self.tickers)
        self._history_close: pd.DataFrame | None = None
        self._sim_cursor: int = 0
        self._staleness: dict[str, bool] = {}

    def download_history(self, period: str = "2y") -> pd.DataFrame:
        """Parallel per-ticker yfinance download; skips delisted / timeouts; rebuilds buffer."""
        wanted = list(self.tickers)
        ok: dict[str, pd.Series] = {}
        max_workers = min(8, max(1, len(wanted)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_download_one_close, t, period): t for t in wanted}
            for fut in as_completed(futs):
                t, series = fut.result()
                if series is not None:
                    ok[t] = series
        if len(ok) < 3:
            return pd.DataFrame()
        closes = pd.DataFrame(ok).sort_index()
        closes = closes.ffill(limit=3).bfill(limit=3)
        closes = closes.dropna(axis=1, how="all")
        self.tickers = list(closes.columns)
        self.buffer = CircularBuffer(self.settings.lookback_days, self.tickers)
        self._staleness = {}
        for c in closes.columns:
            last = closes[c].last_valid_index()
            if last is not None:
                gap = (closes.index[-1] - last).days
                self._staleness[str(c)] = gap > 2
        self._history_close = closes
        mat = closes.values.astype(np.float64)
        valid = np.isfinite(mat).all(axis=1)
        mat = mat[valid]
        idx = closes.index[valid]
        if len(idx) == 0:
            return closes
        self.buffer.load_initial(mat, idx)
        self._sim_cursor = max(0, len(closes) - 1)
        return closes

    async def fetch_batch_async(self, period: str = "2y") -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.download_history(period))

    def simulation_step(self) -> pd.DataFrame:
        """Replay history: rolling window ending at cursor, optional lognormal noise (brief §2.2)."""
        if self._history_close is None or self._history_close.empty:
            return self.buffer.get_closes_df()
        hist = self._history_close
        n = len(hist)
        if n == 0:
            return self.buffer.get_closes_df()
        step = int(max(1, self.settings.simulation_speed))
        self._sim_cursor = (self._sim_cursor + step) % n
        end = min(self._sim_cursor + 1, n)
        start = max(0, end - self.settings.lookback_days)
        window = hist.iloc[start:end].copy()
        mat = window.values.astype(np.float64)
        if self.settings.simulation_noise_std > 0:
            mat = mat * np.exp(np.random.randn(*mat.shape) * self.settings.simulation_noise_std)
        self.buffer.load_initial(mat, window.index)
        return self.buffer.get_closes_df()

    def latest_closes(self) -> pd.DataFrame:
        return self.buffer.get_closes_df()
