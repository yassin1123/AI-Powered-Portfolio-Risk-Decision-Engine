"""JSON risk reports (brief §11); optional PDF stub."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class RiskReport:
    report_id: str
    generated_at: str
    cycle_ms: float
    payload: dict[str, Any]


def basel_traffic_light(actual_returns: np.ndarray, var_99: np.ndarray) -> str:
    """Basel II traffic light: breach when actual return worse than -VaR(99%) (brief §10.2)."""
    n = min(len(actual_returns), len(var_99), 252)
    if n < 10:
        return "GREEN"
    r, v = actual_returns[-n:], var_99[-n:]
    breaches = int(np.sum(r < -v))
    if breaches <= 4:
        return "GREEN"
    if breaches <= 9:
        return "YELLOW"
    return "RED"


def build_json_report(
    cycle_ms: float,
    portfolio: dict[str, Any],
    var_block: dict[str, Any],
    volatility: dict[str, Any],
    correlation: dict[str, Any],
    risk_contributions: dict[str, float],
    anomalies: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    stress_tests: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_id": str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cycle_ms": cycle_ms,
        "portfolio": portfolio,
        "var": var_block,
        "volatility": volatility,
        "correlation": correlation,
        "risk_contributions": risk_contributions,
        "anomalies": anomalies,
        "signals": signals,
        "stress_tests": stress_tests,
    }


def build_pdf_report(payload: dict[str, Any], path: str) -> None:
    """Minimal PDF export via ReportLab (archival)."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return
    c = canvas.Canvas(path, pagesize=letter)
    c.drawString(72, 720, f"Risk Report {payload.get('report_id', '')}")
    c.drawString(72, 700, str(payload.get("generated_at", "")))
    c.showPage()
    c.save()
