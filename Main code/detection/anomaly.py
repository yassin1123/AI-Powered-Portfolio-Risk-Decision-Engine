"""Five-layer anomaly detection + conjunction rule (brief §4)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import pinv

from features.returns import FeatureBundle
from pre.settings import AppSettings

Layer = Literal["ZSCORE", "CUSUM", "MAHAL", "VR", "DRAWDOWN"]
Severity = Literal["WATCH", "WARNING", "CRITICAL"]


@dataclass
class AnomalyEvent:
    event_id: str
    timestamp: str
    layer: Layer
    severity: Severity
    assets: list[str]
    metric_value: float
    threshold: float
    attribution: dict[str, float]
    conjunction_count: int
    recommended_action: str


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def zscore_layer(
    lr: pd.DataFrame, w: pd.Series, cfg: AppSettings
) -> list[AnomalyEvent]:
    out: list[AnomalyEvent] = []
    win = cfg.anomaly.z_window
    last = lr.iloc[-win:]
    mu = last.mean()
    sig = last.std(ddof=1)
    last_r = lr.iloc[-1]
    for t in lr.columns:
        if pd.isna(sig.get(t)) or sig.get(t, 0) == 0:
            continue
        z = float((last_r[t] - mu[t]) / sig[t])
        az = abs(z)
        if az <= 2.0:
            continue
        if az > 4.0:
            sev: Severity = "CRITICAL"
            th = 4.0
        elif az > 3.0:
            sev = "WARNING"
            th = 3.0
        else:
            sev = "WATCH"
            th = 2.0
        out.append(
            AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=_iso_now(),
                layer="ZSCORE",
                severity=sev,
                assets=[t],
                metric_value=az,
                threshold=th,
                attribution={t: 1.0},
                conjunction_count=1,
                recommended_action="MONITOR",
            )
        )
    # portfolio z
    w = w.reindex(lr.columns).fillna(0.0)
    pr = (lr * w).sum(axis=1)
    pl = pr.iloc[-win:]
    pz = float((pr.iloc[-1] - pl.mean()) / pl.std(ddof=1)) if pl.std(ddof=1) > 0 else 0.0
    apz = abs(pz)
    if apz > 2.0:
        contrib = (w * (last_r - mu)).abs().sort_values(ascending=False).head(3)
        att = (contrib / contrib.sum()).to_dict() if contrib.sum() != 0 else {}
        if apz > 4.0:
            sev = "CRITICAL"
            th = 4.0
        elif apz > 3.0:
            sev = "WARNING"
            th = 3.0
        else:
            sev = "WATCH"
            th = 2.0
        out.append(
            AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=_iso_now(),
                layer="ZSCORE",
                severity=sev,
                assets=list(att.keys()),
                metric_value=apz,
                threshold=th,
                attribution={k: float(v) for k, v in att.items()},
                conjunction_count=1,
                recommended_action="MONITOR",
            )
        )
    return out


def cusum_portfolio_stats(
    returns: np.ndarray,
    mu0: float,
    k: float,
    h: float,
) -> tuple[float, float, int]:
    """
    Two-sided CUSUM with alarm resets (brief §4.2).
    Returns final (sp, sm, n_alarms) after processing the series.
    """
    sp = sm = 0.0
    alarms = 0
    for rv in np.asarray(returns, dtype=float).ravel():
        sp = max(0.0, sp + (rv - mu0 - k))
        sm = max(0.0, sm - rv + mu0 - k)
        if sp > h or sm > h:
            alarms += 1
            sp = sm = 0.0
    return sp, sm, alarms


def cusum_layer(lr: pd.Series, cfg: AppSettings) -> list[AnomalyEvent]:
    """One-sided CUSUM on portfolio returns (brief §4.2)."""
    out: list[AnomalyEvent] = []
    r = lr.dropna().iloc[-120:]
    if len(r) < 30:
        return out
    mu0 = float(r.mean())
    sig = float(r.std(ddof=1))
    if sig <= 0:
        return out
    k = cfg.anomaly.cusum_k_sigma * sig
    h = cfg.anomaly.cusum_h_sigma * sig
    sp, sm, alarms = cusum_portfolio_stats(r.values, mu0, k, h)
    if alarms > 0:
        out.append(
            AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=_iso_now(),
                layer="CUSUM",
                severity="WARNING",
                assets=["PORTFOLIO"],
                metric_value=max(sp, sm),
                threshold=h,
                attribution={"PORTFOLIO": 1.0},
                conjunction_count=1,
                recommended_action="REDUCE_POSITION",
            )
        )
    return out


def lo_macinlay_vr(r: np.ndarray, k: int) -> tuple[float, float]:
    n = len(r)
    if n < k + 20 or k < 2:
        return 1.0, 1.0
    var1 = np.var(r, ddof=1)
    if var1 <= 0:
        return 1.0, 1.0
    rk = pd.Series(r).rolling(k).sum().dropna().values
    var_k = np.var(rk, ddof=1)
    vr = var_k / (k * var1)
    se = np.sqrt((2.0 * (2 * k - 1) * (k - 1)) / (3.0 * k * n)) if n > 0 else 1.0
    return float(vr), float(se)


def variance_ratio_layer(lr: pd.DataFrame, cfg: AppSettings) -> list[AnomalyEvent]:
    out: list[AnomalyEvent] = []
    for col in lr.columns:
        r = lr[col].dropna().values
        for k in (2, 5, 10):
            vr, se = lo_macinlay_vr(r, k)
            if se <= 0:
                continue
            if abs(vr - 1.0) > 2.0 * se:
                out.append(
                    AnomalyEvent(
                        event_id=str(uuid.uuid4()),
                        timestamp=_iso_now(),
                        layer="VR",
                        severity="WARNING",
                        assets=[col],
                        metric_value=vr,
                        threshold=1.0 + 2 * se,
                        attribution={col: 1.0},
                        conjunction_count=1,
                        recommended_action="MONITOR",
                    )
                )
                break
    return out


def mahalanobis_layer(
    lr: pd.DataFrame, cfg: AppSettings
) -> list[AnomalyEvent]:
    out: list[AnomalyEvent] = []
    win = min(252, len(lr))
    if win < 30:
        return out
    x = lr.iloc[-win:].dropna(how="any")
    if len(x) < 30:
        return out
    arr = x.values
    mu = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False)
    pinv_cov = pinv(cov)
    r_last = lr.iloc[-1].reindex(x.columns).values.astype(float)
    diff = r_last - mu
    d2 = float(diff @ pinv_cov @ diff)
    n = arr.shape[1]
    q = cfg.anomaly.mahalanobis_chi2_quantile
    th = float(stats.chi2.ppf(q, df=n))
    if d2 > th:
        out.append(
            AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=_iso_now(),
                layer="MAHAL",
                severity="CRITICAL",
                assets=list(x.columns),
                metric_value=d2,
                threshold=th,
                attribution={c: abs(diff[i]) for i, c in enumerate(x.columns)},
                conjunction_count=1,
                recommended_action="REDUCE_POSITION",
            )
        )
    return out


def drawdown_layer(
    feat: FeatureBundle, w: pd.Series, cfg: AppSettings
) -> list[AnomalyEvent]:
    out: list[AnomalyEvent] = []
    dd = feat.drawdown_asset.iloc[-1]
    for t, v in dd.items():
        if pd.isna(v):
            continue
        av = abs(float(v))
        if av >= cfg.anomaly.drawdown_critical:
            sev: Severity = "CRITICAL"
            th = cfg.anomaly.drawdown_critical
        elif av >= cfg.anomaly.drawdown_warning:
            sev = "WARNING"
            th = cfg.anomaly.drawdown_warning
        elif av >= cfg.anomaly.drawdown_watch:
            sev = "WATCH"
            th = cfg.anomaly.drawdown_watch
        else:
            continue
        out.append(
            AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=_iso_now(),
                layer="DRAWDOWN",
                severity=sev,
                assets=[str(t)],
                metric_value=av,
                threshold=th,
                attribution={str(t): 1.0},
                conjunction_count=1,
                recommended_action="REDUCE_POSITION",
            )
        )
    pdd = feat.drawdown_portfolio.iloc[-1]
    if not pd.isna(pdd) and abs(float(pdd)) >= cfg.anomaly.drawdown_warning:
        out.append(
            AnomalyEvent(
                event_id=str(uuid.uuid4()),
                timestamp=_iso_now(),
                layer="DRAWDOWN",
                severity="CRITICAL" if abs(float(pdd)) >= cfg.anomaly.drawdown_critical else "WARNING",
                assets=["PORTFOLIO"],
                metric_value=abs(float(pdd)),
                threshold=cfg.anomaly.drawdown_critical,
                attribution={"PORTFOLIO": 1.0},
                conjunction_count=1,
                recommended_action="REDUCE_POSITION",
            )
        )
    return out


def _event_to_dict(e: AnomalyEvent) -> dict[str, Any]:
    return {
        "event_id": e.event_id,
        "timestamp": e.timestamp,
        "layer": e.layer,
        "severity": e.severity,
        "assets": e.assets,
        "metric_value": e.metric_value,
        "threshold": e.threshold,
        "attribution": e.attribution,
        "conjunction_count": e.conjunction_count,
        "recommended_action": e.recommended_action,
    }


def apply_conjunction(
    events: list[AnomalyEvent], min_layers: int
) -> list[AnomalyEvent]:
    """Keep events where >= min_layers distinct methods fired for same primary asset (brief §4)."""
    by_asset: dict[str, set[str]] = {}
    for e in events:
        key = e.assets[0] if e.assets else "PORTFOLIO"
        by_asset.setdefault(key, set()).add(e.layer)
    out: list[AnomalyEvent] = []
    for e in events:
        key = e.assets[0] if e.assets else "PORTFOLIO"
        cnt = len(by_asset.get(key, set()))
        if cnt >= min_layers or e.severity == "CRITICAL":
            out.append(
                AnomalyEvent(
                    event_id=e.event_id,
                    timestamp=e.timestamp,
                    layer=e.layer,
                    severity=e.severity,
                    assets=e.assets,
                    metric_value=e.metric_value,
                    threshold=e.threshold,
                    attribution=e.attribution,
                    conjunction_count=cnt,
                    recommended_action=e.recommended_action,
                )
            )
    return out


class AnomalyPipeline:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def run(self, feat: FeatureBundle, weights: pd.Series) -> list[dict[str, Any]]:
        lr = feat.log_returns_1d
        w = weights.reindex(lr.columns).fillna(0.0)
        ev: list[AnomalyEvent] = []
        ev.extend(zscore_layer(lr, w, self.settings))
        pr = (lr * w).sum(axis=1)
        ev.extend(cusum_layer(pr, self.settings))
        ev.extend(mahalanobis_layer(lr, self.settings))
        ev.extend(variance_ratio_layer(lr, self.settings))
        ev.extend(drawdown_layer(feat, w, self.settings))
        conj = apply_conjunction(ev, self.settings.anomaly.conjunction_min_layers)
        return [_event_to_dict(e) for e in conj][-20:]
