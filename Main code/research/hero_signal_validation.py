"""Hero correlation-z predictive content: buckets vs forward vol / drawdown / avg corr (elite brief §4.3, §5.4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.corr_z_path import correlation_z_series
from pre.settings import load_settings


def _cov_avg_corr(ret_window: np.ndarray) -> float:
    """Average pairwise correlation of columns (assets x days), minimum 2 assets."""
    if ret_window.shape[1] < 2 or ret_window.shape[0] < 2:
        return float("nan")
    c = np.corrcoef(ret_window.T)
    n = c.shape[0]
    if n < 2:
        return float("nan")
    return float((np.sum(c) - np.trace(c)) / (n * (n - 1)))


def build_validation_table(
    closes: pd.DataFrame,
    settings,
    *,
    horizons: tuple[int, ...] = (5, 10, 20),
    warmup: int = 130,
) -> pd.DataFrame:
    z = correlation_z_series(closes, settings, warmup=warmup)
    tickers = list(closes.columns)
    lr = np.log(closes / closes.shift(1))
    rows: list[dict[str, float | str]] = []

    for dt in z.index:
        try:
            ti = closes.index.get_loc(dt)
        except KeyError:
            continue
        if isinstance(ti, slice):
            ti = ti.start
        zi = float(z.loc[dt])
        q = z.dropna()
        pct = float((q <= zi).mean()) if len(q) else 0.5
        if pct <= 0.25:
            bucket = "low"
        elif pct <= 0.5:
            bucket = "medium_low"
        elif pct <= 0.75:
            bucket = "medium_high"
        else:
            bucket = "extreme_high"

        rec: dict[str, float | str] = {"date": str(dt.date()), "corr_z": zi, "bucket": bucket}
        for h in horizons:
            sl = slice(ti + 1, ti + 1 + h)
            if ti + 1 + h > len(closes):
                continue
            fw = lr.iloc[sl][tickers].dropna(how="any")
            if len(fw) < max(3, h // 2):
                continue
            rv = fw.values.astype(float)
            fused = rv.ravel()
            fused = fused[np.isfinite(fused)]
            if len(fused) < 2:
                continue
            rec[f"fwd_vol_{h}d_ann"] = float(np.std(fused, ddof=1) * np.sqrt(252))
            w = np.ones(rv.shape[1]) / rv.shape[1]
            port_lr = (rv @ w).ravel()
            eq = np.exp(np.cumsum(port_lr))
            peak = np.maximum.accumulate(eq)
            dd = eq / np.clip(peak, 1e-12, None) - 1.0
            rec[f"fwd_mdd_{h}d"] = float(dd.min())
            rec[f"fwd_avg_corr_{h}d"] = _cov_avg_corr(rv)
        rows.append(rec)

    return pd.DataFrame(rows)


def bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "bucket" not in df.columns:
        return pd.DataFrame()
    num_cols = [c for c in df.columns if c.startswith("fwd_") and df[c].dtype != object]
    return df.groupby("bucket", observed=True)[num_cols].mean().reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description="Hero signal validation tables")
    p.add_argument("--panel", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    root = ROOT
    cfg = args.config or (root / "config.yaml")
    settings = load_settings(cfg)
    out_dir = root / "research" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        from backtest.run import synthetic_closes

        closes = synthetic_closes(
            daily_drift=settings.backtest.synthetic_daily_drift,
            vol_per_bar=settings.backtest.synthetic_vol_per_bar,
        )
    elif args.panel:
        closes = pd.read_csv(args.panel, index_col=0, parse_dates=True)
        closes.index = pd.DatetimeIndex(pd.to_datetime(closes.index))
    else:
        from data.panel_store import load_processed_closes

        closes, _ = load_processed_closes(root)
        if closes.empty:
            print("No panel; use --synthetic or build_data_panel / --panel", file=sys.stderr)
            sys.exit(1)

    tbl = build_validation_table(closes, settings)
    tbl.to_csv(out_dir / "hero_signal_validation_series.csv", index=False)
    summ = bucket_summary(tbl)
    summ.to_csv(out_dir / "hero_signal_validation_buckets.csv", index=False)
    print(f"Wrote {out_dir / 'hero_signal_validation_series.csv'}")
    print(f"Wrote {out_dir / 'hero_signal_validation_buckets.csv'}")


if __name__ == "__main__":
    main()
