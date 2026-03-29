"""Plotly Dash dashboard — LIVE + RESEARCH (Dashboard UI Engineering Brief). Read-only: core.publish.read_snapshot."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from core.publish import read_snapshot


def _header_daily_pct(h: dict) -> float:
    """Simple return % from header (pipeline sends log-return sum + optional expm1 field)."""
    if h.get("daily_return_simple_approx") is not None:
        return float(h["daily_return_simple_approx"]) * 100.0
    return float(np.expm1(float(h.get("daily_return", 0.0)))) * 100.0


def _path_is_research(pathname: str | None) -> bool:
    p = (pathname or "/").lower().strip()
    return p.rstrip("/") == "/research" or p.endswith("/research")

_ROOT = Path(__file__).resolve().parent.parent
_RESEARCH_OUT = _ROOT / "research" / "outputs"
_RESEARCH_FIG = _ROOT / "research" / "figures" / "killer_overlay.png"

# Rebuild research tab only when output files change (navigation was re-reading CSVs + PNG every time).
_research_page_cache: dict[str, object | None] = {"sig": None, "root": None}

_WRAP_TRANSITION = {
    "transition": "opacity 0.18s ease-out",
    "willChange": "opacity",
}

DARK = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
AMBER = "#d29922"
RED = "#f85149"
BLUE = "#58a6ff"

REGIME_BORDER = {
    "CALM": GREEN,
    "NORMAL": GREEN,
    "TRANSITION": AMBER,
    "STRESS": AMBER,
    "CRISIS": RED,
}


def _regime_color(regime: str) -> str:
    r = (regime or "").upper()
    for k, v in REGIME_BORDER.items():
        if k in r or r in k:
            return v
    return ACCENT


_CORR_Z_SPIKE = 1.5


def _regime_fill_rgba(reg: str) -> str:
    u = (reg or "").upper()
    if "STRESS" in u or "CRISIS" in u:
        return "rgba(248,81,73,0.14)"
    if "TRANSITION" in u:
        return "rgba(210,153,34,0.14)"
    return "rgba(63,185,80,0.10)"


def _action_line(ss: dict) -> str:
    dec = ss.get("decision") or {}
    pct = float(dec.get("exposure_scale") or 1.0) * 100
    hedge = bool(dec.get("activate_hedge"))
    pr = str(dec.get("priority") or ss.get("decision_priority") or "normal")
    if pr in ("normal", "diversification_regime", "signals_only_neutral") and pct >= 99 and not hedge:
        return "→ Action: Maintain standard risk budget; no mandatory de-risk or hedge activation."
    parts = [f"Scale target exposure to ~{pct:.0f}% of baseline"]
    if hedge:
        parts.append("activate hedge overlay")
    return "→ Action: " + " + ".join(parts) + "."


def _decision_explanation_text(ss: dict) -> str:
    dec = ss.get("decision") or {}
    pr = str(dec.get("priority") or ss.get("decision_priority") or "normal")
    cz = float(ss.get("corr_z") or 0.0)
    reg = str(ss.get("regime") or "")
    bucket = str(dec.get("corr_bucket") or ss.get("corr_bucket") or "")
    lines = {
        "stress_corr_override": (
            "Stressed regime coincides with very high correlation instability. "
            "That combination usually means systemic co-movement and fragile diversification. "
            "The engine overrides discretionary risk-taking, cuts exposure sharply, and prioritises defensive positioning."
        ),
        "corr_crisis": (
            "Correlation Instability (Z-score) has spiked above the crisis threshold. "
            "That indicates co-movement is unusually high versus its own history—diversification is doing less work when you need it most. "
            "The system scales risk down and turns hedging on to limit drawdowns."
        ),
        "anomaly_suppress": (
            "Multiple independent anomaly detectors fired together. "
            "That suggests the return distribution or structure of the book may be shifting. "
            "The system suppresses non-defensive risk until the picture stabilises."
        ),
        "stressed_regime": (
            "The regime classifier labels conditions as stressed (vol and/or correlation elevated). "
            "Exposure is reduced and hedges may be armed depending on the correlation bucket."
        ),
        "transition": (
            "Markets are in a transition regime—signals are noisier and correlations less stable. "
            "The engine dials gross exposure back until the state becomes clearer."
        ),
        "var_breach_risk": (
            "Estimated tail loss (VaR) is elevated versus your configured limit. "
            "The system trims risk budget before a larger shock crystallises."
        ),
        "diversification_regime": (
            "Correlation instability is unusually low—dispersion and diversification are more available. "
            "The engine can allow slightly fuller risk within constraints."
        ),
        "normal": (
            "No stress override is active: correlation instability, regime, anomalies, and VaR are not jointly breaching aggressive thresholds. "
            "The book follows the standard signal and risk-budget path."
        ),
        "signals_only_neutral": (
            "Decision layer is neutral (signals-only backtest mode); no regime-based de-risking is applied here."
        ),
    }
    body = lines.get(pr, lines["normal"])
    extra = []
    if cz > _CORR_Z_SPIKE:
        extra.append(f"Current Z-score {cz:.2f} supports a correlation-focused read.")
    if reg and pr != "normal":
        extra.append(f"Regime label: {reg}.")
    if bucket and bucket != "none":
        extra.append(f"Correlation bucket: {bucket}.")
    tail = " " + " ".join(extra) if extra else ""
    return body + tail


def _killer_strip_labels(ss: dict, vp: dict) -> tuple[list[tuple[str, str, str]], str]:
    """(label, value, color) x4 and strip border color."""
    cz = float(ss.get("corr_z") or 0.0)
    if abs(cz) >= _CORR_Z_SPIKE:
        corr_l, corr_v, corr_c = "Correlation", f"HIGH ({cz:.2f} ↑)", RED
    elif abs(cz) >= 1.0:
        corr_l, corr_v, corr_c = "Correlation", "MODERATE", AMBER
    else:
        corr_l, corr_v, corr_c = "Correlation", "NORMAL", GREEN

    pred = float(ss.get("predicted_ann_vol_pct") or 0.0)
    tgt = float(ss.get("target_ann_vol_pct") or 10.0)
    if pred > tgt * 1.25:
        vol_l, vol_v, vol_c = "Volatility", "HIGH", RED
    elif pred > tgt * 1.08:
        vol_l, vol_v, vol_c = "Volatility", "MODERATE", AMBER
    else:
        vol_l, vol_v, vol_c = "Volatility", "MODERATE", GREEN if pred < tgt else AMBER

    reg = str(ss.get("regime") or "?")
    reg_u = reg.upper()
    if any(x in reg_u for x in ("STRESS", "CRISIS")):
        rg_l, rg_v, rg_c = "Regime", reg, RED
    elif "TRANSITION" in reg_u:
        rg_l, rg_v, rg_c = "Regime", reg, AMBER
    else:
        rg_l, rg_v, rg_c = "Regime", reg, GREEN

    pr = str(ss.get("decision_priority") or (ss.get("decision") or {}).get("priority") or "normal")
    if pr in ("normal", "diversification_regime", "signals_only_neutral") and abs(cz) < _CORR_Z_SPIKE:
        risk_l, risk_v, risk_c = "Risk level", "NORMAL", GREEN
    elif pr in ("transition", "var_breach_risk"):
        risk_l, risk_v, risk_c = "Risk level", "ELEVATED", AMBER
    else:
        risk_l, risk_v, risk_c = "Risk level", "ELEVATED", RED

    strip_border = risk_c
    return (
        [
            (corr_l, corr_v, corr_c),
            (vol_l, vol_v, vol_c),
            (rg_l, rg_v, rg_c),
            (risk_l, risk_v, risk_c),
        ],
        strip_border,
    )


def _strip_chip(label: str, value: str, color: str) -> html.Div:
    return html.Div(
        style={
            "flex": "1 1 140px",
            "minWidth": "120px",
            "padding": "10px 12px",
            "borderRadius": "8px",
            "border": f"2px solid {color}",
            "backgroundColor": "#161b22",
        },
        children=[
            html.Div(f"📊 {label}", style={"fontSize": "10px", "color": MUTED, "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "15px", "fontWeight": "700", "color": color, "marginTop": "4px"}),
        ],
    )


def _fig_killer_overlay(ov: dict) -> go.Figure:
    dd = ov.get("drawdown") or []
    cz = ov.get("corr_z") or []
    reg = ov.get("regime") or []
    n = min(len(dd), len(cz), len(reg), 500)
    if n < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Live history builds as the risk loop runs…",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=14),
        )
        fig.update_layout(
            title="📉 Drawdown vs Correlation Instability (regime shading)",
            paper_bgcolor=PANEL,
            plot_bgcolor=DARK,
            font_color=TEXT,
            margin=dict(t=48, b=40, l=55, r=55),
        )
        return fig

    dd = [float(x) * 100 for x in dd[-n:]]
    cz = [float(x) for x in cz[-n:]]
    reg = [str(x) for x in reg[-n:]]
    x = list(range(n))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    shapes: list[dict] = []
    i0 = 0
    for i in range(1, n + 1):
        new_seg = i == n or reg[i] != reg[i0]
        if not new_seg:
            continue
        i1 = i - 1
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=x[i0] - 0.5,
                x1=x[i1] + 0.5,
                y0=0,
                y1=1,
                fillcolor=_regime_fill_rgba(reg[i0]),
                line_width=0,
                layer="below",
            )
        )
        i0 = i if i < n else i0

    fig.update_layout(shapes=shapes)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=dd,
            mode="lines",
            name="Portfolio drawdown",
            line=dict(color="#58a6ff", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cz,
            mode="lines",
            name="Correlation Instability (Z-score)",
            line=dict(color="#ffa657", width=2),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="📉 Drawdown vs Correlation Instability (regime shading)",
        paper_bgcolor=PANEL,
        plot_bgcolor=DARK,
        font_color=TEXT,
        hovermode="x unified",
        margin=dict(t=52, b=48, l=58, r=58),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor="#30363d", title="Bars (live history)")
    fig.update_yaxes(title_text="Drawdown %", secondary_y=False, gridcolor="#30363d")
    fig.update_yaxes(title_text="Z-score", secondary_y=True, gridcolor="#30363d")
    return fig


def _signal_color(sig: str) -> str:
    s = (sig or "").upper()
    if "REDUCE" in s:
        return RED
    if "TRANSITION" in s:
        return AMBER
    if "MAINTAIN" in s:
        return GREEN
    if "EXPAND" in s or "ALLOW" in s:
        return ACCENT
    return TEXT


def _build_system_command(ss: dict) -> html.Div:
    tr = ss.get("decision_trace") or {}
    dec = ss.get("decision") or {}
    sig = str(tr.get("system_signal") or dec.get("system_signal") or "MAINTAIN")
    conf = float(tr.get("confidence") if tr.get("confidence") is not None else dec.get("confidence") or 0.55)
    reg = str(ss.get("regime") or "?")
    rule = str(tr.get("winning_rule_id") or dec.get("winning_rule_id") or "—")
    mult = float(dec.get("exposure_scale") or 1.0)
    col = _signal_color(sig)
    action = _action_line(ss)
    return html.Div(
        style={
            "backgroundColor": "#0d1117",
            "border": f"4px solid {col}",
            "borderRadius": "12px",
            "padding": "22px 26px",
            "marginBottom": "16px",
        },
        children=[
            html.Div(
                "SYSTEM OUTPUT — control layer",
                style={
                    "fontSize": "11px",
                    "color": MUTED,
                    "letterSpacing": "0.14em",
                    "fontWeight": "700",
                },
            ),
            html.Div(
                f"SYSTEM SIGNAL: {sig}",
                style={
                    "fontSize": "clamp(22px, 4vw, 32px)",
                    "fontWeight": "800",
                    "color": col,
                    "marginTop": "10px",
                    "letterSpacing": "0.03em",
                },
            ),
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "22px", "marginTop": "16px"},
                children=[
                    html.Div(
                        [
                            html.Div("CONFIDENCE", style={"fontSize": "10px", "color": MUTED}),
                            html.Div(f"{conf * 100:.0f}%", style={"fontSize": "24px", "fontWeight": "700"}),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div("REGIME", style={"fontSize": "10px", "color": MUTED}),
                            html.Div(reg, style={"fontSize": "24px", "fontWeight": "700", "color": TEXT}),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div("RISK MULTIPLIER", style={"fontSize": "10px", "color": MUTED}),
                            html.Div(f"{mult:.2f}×", style={"fontSize": "24px", "fontWeight": "700", "color": AMBER}),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div("WINNING RULE", style={"fontSize": "10px", "color": MUTED}),
                            html.Div(rule, style={"fontSize": "13px", "fontWeight": "600", "color": MUTED, "maxWidth": "280px"}),
                        ]
                    ),
                ],
            ),
            html.Div(
                ["→ ", action],
                style={"marginTop": "14px", "fontSize": "15px", "fontWeight": "600", "color": AMBER},
            ),
        ],
    )


def _build_metrics_institutional(ss: dict, vp: dict, corr: dict) -> html.Div:
    pct = vp.get("var_99_percentile_vs_history")
    pct_s = f"{pct:.0f}th" if pct is not None and isinstance(pct, (int, float)) else "—"
    br = int(vp.get("breaches_30d") or 0)
    trend = str(vp.get("var_trend_label") or "—")
    div_s = float(corr.get("diversification_score") or 0.0)
    div_note = str(corr.get("diversification_note") or "")
    chips = [
        _strip_chip("Diversification", f"{div_s:.2f} (1−ρ̄)", GREEN if div_s > 0.45 else AMBER),
        _strip_chip("VaR pctile", pct_s, AMBER if pct is not None and pct > 75 else GREEN),
        _strip_chip("Breaches (30d)", str(br), RED if br >= 3 else GREEN),
        _strip_chip("VaR trend", trend.upper(), RED if trend == "increasing" else GREEN),
    ]
    return html.Div(
        style={
            "marginBottom": "14px",
            "padding": "12px 14px",
            "borderRadius": "8px",
            "backgroundColor": "#161b22",
            "borderLeft": f"4px solid {ACCENT}",
        },
        children=[
            html.Div(
                "⚡ Risk metrics — institutional view",
                style={
                    "fontSize": "11px",
                    "color": MUTED,
                    "marginBottom": "10px",
                    "textTransform": "uppercase",
                    "fontWeight": "600",
                },
            ),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}, children=chips),
            html.Div(div_note, style={"fontSize": "12px", "color": MUTED, "marginTop": "10px", "maxWidth": "900px"}),
        ],
    )


def _build_strip(ss: dict, vp: dict) -> html.Div:
    chips, border = _killer_strip_labels(ss, vp)
    row = [_strip_chip(l, v, c) for l, v, c in chips]
    return html.Div(
        style={
            "marginBottom": "14px",
            "padding": "12px 14px",
            "borderRadius": "8px",
            "backgroundColor": "#161b22",
            "borderLeft": f"4px solid {border}",
        },
        children=[
            html.Div(
                "📊 Current market structure",
                style={
                    "fontSize": "11px",
                    "color": MUTED,
                    "marginBottom": "10px",
                    "textTransform": "uppercase",
                    "fontWeight": "600",
                },
            ),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}, children=row),
        ],
    )


def _build_explain(ss: dict) -> html.Div:
    dec = ss.get("decision") or {}
    codes = dec.get("codes") or []
    pr = dec.get("priority") or ss.get("decision_priority") or "—"
    tr = ss.get("decision_trace") or {}
    if tr:
        mech = [html.Li(line, style={"marginBottom": "6px"}) for line in (tr.get("mechanical_lines") or [])]
        drv = [html.Li(line, style={"marginBottom": "4px", "fontSize": "13px", "color": MUTED}) for line in (tr.get("driver_lines") or [])]
        pos_lines = tr.get("positioning", {}).get("lines") or []
        pos_ul = [html.Li(p, style={"marginBottom": "4px"}) for p in pos_lines]
        flags = tr.get("condition_flags") or {}
        flag_rows = [html.Div(f"{k}: {v}", style={"fontSize": "11px", "color": MUTED}) for k, v in flags.items()]
        conc = tr.get("conclusion") or ""
        policy_note = tr.get("policy_note") or ""
        children = [
            html.Div(
                "Decision drivers (mechanical)",
                style={"fontSize": "11px", "color": MUTED, "textTransform": "uppercase", "marginBottom": "8px"},
            ),
            html.Ul(mech, style={"marginTop": "0", "lineHeight": "1.5", "paddingLeft": "20px"}),
            html.Div(
                "Score mass (audit)",
                style={"fontSize": "11px", "color": MUTED, "textTransform": "uppercase", "margin": "14px 0 6px"},
            ),
            html.Ul(drv, style={"marginTop": "0", "paddingLeft": "20px"}),
            html.Div(
                "Positioning",
                style={"fontSize": "11px", "color": MUTED, "textTransform": "uppercase", "margin": "14px 0 6px"},
            ),
            html.Ul(pos_ul, style={"marginTop": "0", "paddingLeft": "20px"}),
            html.Div(
                conc,
                style={
                    "marginTop": "14px",
                    "padding": "12px",
                    "backgroundColor": "#21262d",
                    "borderRadius": "6px",
                    "fontWeight": "600",
                    "fontSize": "14px",
                    "lineHeight": "1.5",
                },
            ),
            html.Details(
                style={"marginTop": "12px"},
                children=[
                    html.Summary("Rule condition flags & policy", style={"cursor": "pointer", "color": MUTED, "fontSize": "12px"}),
                    html.Div(
                        [html.P(policy_note, style={"fontSize": "12px", "color": TEXT, "marginTop": "8px"})]
                        + flag_rows
                        + [
                            html.Div(
                                f"Priority: {pr} · codes: {', '.join(str(c) for c in codes) or '—'}",
                                style={"fontSize": "12px", "color": MUTED, "marginTop": "8px"},
                            )
                        ],
                    ),
                ],
            ),
        ]
    else:
        txt = _decision_explanation_text(ss)
        children = [
            html.Div(
                "🧠 Decision explanation",
                style={"fontSize": "11px", "color": MUTED, "textTransform": "uppercase", "marginBottom": "8px"},
            ),
            html.P(txt, style={"fontSize": "14px", "lineHeight": "1.55", "margin": "0"}),
            html.Details(
                style={"marginTop": "12px"},
                children=[
                    html.Summary("Technical codes", style={"cursor": "pointer", "color": MUTED, "fontSize": "12px"}),
                    html.Div(
                        f"Priority: {pr} · codes: {', '.join(str(c) for c in codes) or '—'}",
                        style={"fontSize": "12px", "color": MUTED, "marginTop": "6px"},
                    ),
                ],
            ),
        ]
    return html.Div(
        style={
            "backgroundColor": PANEL,
            "borderRadius": "8px",
            "padding": "16px 18px",
            "marginBottom": "14px",
            "borderLeft": f"4px solid {ACCENT}",
        },
        children=[
            html.Div(
                "🧠 Decision trace",
                style={"fontSize": "11px", "color": MUTED, "textTransform": "uppercase", "marginBottom": "8px"},
            ),
            *children,
        ],
    )


def _severity_rank(s: str) -> int:
    return {"CRITICAL": 0, "WARNING": 1, "WATCH": 2}.get(s or "", 9)


def _severity_border(sev: str) -> str:
    if sev == "CRITICAL":
        return RED
    if sev == "WARNING":
        return AMBER
    return BLUE


def _chip(label: str, value: str, *, accent: bool = False) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "#21262d",
            "border": "1px solid #30363d",
            "borderRadius": "6px",
            "padding": "8px 12px",
            "minWidth": "88px",
        },
        children=[
            html.Div(label, style={"fontSize": "10px", "color": MUTED, "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "14px", "fontWeight": "600", "color": ACCENT if accent else TEXT}),
        ],
    )


def _nav() -> html.Div:
    return html.Div(
        style={
            "display": "flex",
            "gap": "16px",
            "alignItems": "center",
            "marginBottom": "14px",
            "paddingBottom": "10px",
            "borderBottom": "1px solid #30363d",
        },
        children=[
            html.Span("Risk cockpit", style={"fontWeight": "700", "fontSize": "18px", "marginRight": "12px"}),
            dcc.Link("LIVE", href="/", style={"color": ACCENT, "textDecoration": "none", "fontWeight": "600"}),
            dcc.Link("RESEARCH", href="/research", style={"color": MUTED, "textDecoration": "none"}),
        ],
    )


def _fig_var_grid(vp: dict, risk_limit: float) -> go.Figure:
    cats = ["1d 95%", "1d 99%", "10d 95%", "10d 99%"]
    hs = [
        vp.get("hs_var_95_1d", 0) * 100,
        vp.get("hs_var_99_1d", 0) * 100,
        vp.get("hs_var_95_10d", 0) * 100,
        vp.get("hs_var_99_10d", 0) * 100,
    ]
    mc = [
        vp.get("mc_var_95_1d", 0) * 100,
        vp.get("mc_var_99_1d", 0) * 100,
        vp.get("mc_var_95_10d", 0) * 100,
        vp.get("mc_var_99_10d", 0) * 100,
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Historical sim", x=cats, y=hs, marker_color="#238636"))
    fig.add_trace(go.Bar(name="Monte Carlo", x=cats, y=mc, marker_color="#8957e5"))
    lim_pct = risk_limit * 100
    fig.add_hline(
        y=lim_pct,
        line_dash="dash",
        line_color=AMBER,
        annotation_text=f"Limit 99% 1d ≈ {lim_pct:.2f}%",
        annotation_position="top right",
    )
    zone = vp.get("backtesting_zone") or "—"
    tm = vp.get("tail_multiplier")
    cf = vp.get("cf_cvar_99")
    sub = (
        f"Basel zone: {zone} · CF tail mult: {tm:.2f} · CF CVaR99: {float(cf or 0)*100:.2f}% · "
        "HS = empirical portfolio returns; MC = Gaussian with DCC–GARCH Σ (often lower in calm regimes)"
        if tm is not None
        else f"Basel zone: {zone}"
    )
    fig.update_layout(
        title="⚠️ VaR comparison — historical sim vs Monte Carlo (% loss)",
        barmode="group",
        paper_bgcolor=PANEL,
        plot_bgcolor=DARK,
        font_color=TEXT,
        margin=dict(l=48, r=20, t=52, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        annotations=[
            dict(
                text=sub,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.22,
                showarrow=False,
                font=dict(size=11, color=MUTED),
                xanchor="left",
            )
        ],
    )
    fig.update_yaxes(gridcolor="#30363d", title="%")
    return fig


def _fig_var_trend(vp: dict) -> go.Figure:
    ser = vp.get("var_99_series") or []
    fig = go.Figure()
    if len(ser) >= 2:
        y = [float(v) * 100 for v in ser]
        x = list(range(len(y)))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="MC VaR 99% 1d",
                line=dict(color="#8957e5", width=2),
            )
        )
    else:
        fig.add_annotation(
            text="VaR history builds as the loop runs…",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=13),
        )
    pct = vp.get("var_99_percentile_vs_history")
    br = vp.get("breaches_30d")
    tr = vp.get("var_trend_label")
    parts: list[str] = []
    if pct is not None and isinstance(pct, (int, float)):
        parts.append(f"Percentile vs history: {float(pct):.0f}th")
    parts.append(f"30d breaches: {br}")
    parts.append(f"trend: {tr}")
    sub = " · ".join(parts)
    fig.update_layout(
        title="VaR 99% (MC, 1d) — path & context",
        paper_bgcolor=PANEL,
        plot_bgcolor=DARK,
        font_color=TEXT,
        margin=dict(l=48, r=20, t=52, b=52),
        annotations=[
            dict(
                text=sub,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.2,
                showarrow=False,
                font=dict(size=11, color=MUTED),
                xanchor="left",
            )
        ],
    )
    fig.update_xaxes(gridcolor="#30363d", title="Bars (recent)")
    fig.update_yaxes(gridcolor="#30363d", title="VaR 99 %")
    return fig


def _fig_mc(sims: list, vp: dict) -> go.Figure:
    fig = go.Figure()
    if sims:
        arr = np.asarray(sims, dtype=float)
        arr = arr[np.isfinite(arr)]
        fig.add_trace(go.Histogram(x=arr, nbinsx=55, name="Simulated 1d returns", marker_color="#8957e5", opacity=0.85))
        v95, v99 = vp.get("mc_var_95_1d", 0), vp.get("mc_var_99_1d", 0)
        fig.add_vline(x=-v95, line_color=GREEN, line_width=2, annotation_text="VaR95", annotation_position="top")
        fig.add_vline(x=-v99, line_color=AMBER, line_width=2, annotation_text="VaR99", annotation_position="top")
    fig.update_layout(
        title="Simulated 1-Day Portfolio Returns (MC) + VaR lines",
        paper_bgcolor=PANEL,
        plot_bgcolor=DARK,
        font_color=TEXT,
        margin=dict(l=48, r=20, t=48, b=40),
        bargap=0.05,
    )
    fig.update_xaxes(gridcolor="#30363d", title="P&L (log-return approx.)")
    fig.update_yaxes(gridcolor="#30363d")
    return fig


def _fig_corr(corr: dict) -> go.Figure:
    mat = corr.get("matrix")
    tks = corr.get("tickers") or []
    fig = go.Figure()
    if mat is not None and len(tks) > 0:
        fig.add_trace(
            go.Heatmap(
                z=mat,
                x=tks,
                y=tks,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
            )
        )
    ci = corr.get("contagion_index")
    cz = corr.get("corr_z")
    ds = corr.get("diversification_score")
    sub = (
        f"Contagion: {ci} · Z-score: {cz} · diversification (1−ρ̄): {ds} · "
        f"regime: {corr.get('regime', '—')}"
    )
    fig.update_layout(
        title="🔗 Correlation structure (DCC)",
        paper_bgcolor=PANEL,
        plot_bgcolor=DARK,
        font_color=TEXT,
        margin=dict(l=60, r=20, t=44, b=60),
        height=340,
        annotations=[
            dict(text=sub, xref="paper", yref="paper", x=0, y=1.07, showarrow=False, font=dict(size=11, color=MUTED), xanchor="left")
        ],
    )
    return fig


def _fig_garch_paths(paths: dict[str, list[float]], max_series: int = 10) -> go.Figure:
    fig = go.Figure()
    cols = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#79c0ff", "#ffa657", "#ff7b72"]
    for i, (t, ser) in enumerate(list(paths.items())[:max_series]):
        if not ser:
            continue
        y = [float(v) * 100 for v in ser]
        fig.add_trace(
            go.Scatter(
                y=y,
                mode="lines",
                name=str(t)[:12],
                line=dict(width=1.2, color=cols[i % len(cols)]),
            )
        )
    fig.update_layout(
        title="GARCH conditional vol paths (% per day, recent window)",
        paper_bgcolor=PANEL,
        plot_bgcolor=DARK,
        font_color=TEXT,
        margin=dict(l=48, r=20, t=48, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        xaxis_title="Bar",
        yaxis_title="σ %",
    )
    fig.update_yaxes(gridcolor="#30363d")
    fig.update_xaxes(gridcolor="#30363d")
    return fig


def _fig_stress(stress: dict) -> go.Figure:
    names = []
    vals = []
    for k, v in stress.items():
        if k == "reverse_target_15pct" or not isinstance(v, dict):
            continue
        names.append(k.replace("_", " "))
        vals.append(float(v.get("portfolio_pnl", 0)) * 100)
    fig = go.Figure(data=[go.Bar(x=names, y=vals, marker_color=["#da3633" if y < 0 else "#238636" for y in vals])])
    fig.update_layout(
        title="Historical stress scenarios (portfolio P&L %)",
        paper_bgcolor=PANEL,
        plot_bgcolor=DARK,
        font_color=TEXT,
        margin=dict(l=48, r=20, t=48, b=120),
        xaxis_tickangle=-35,
    )
    fig.update_yaxes(gridcolor="#30363d")
    return fig


def _fig_weights(w: dict, tw: dict) -> go.Figure:
    keys = sorted(set(w) | set(tw), key=lambda k: max(abs(w.get(k, 0)), abs(tw.get(k, 0))), reverse=True)[:18]
    c = [w.get(k, 0) * 100 for k in keys]
    t = [tw.get(k, 0) * 100 for k in keys]
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(name="Current", y=keys, x=c, orientation="h", marker_color="#58a6ff"))
    fig.add_trace(go.Bar(name="Target", y=keys, x=t, orientation="h", marker_color="#3fb950"))
    fig.update_layout(
        title="Weights drift (% allocation)",
        barmode="group",
        paper_bgcolor=PANEL,
        plot_bgcolor=DARK,
        font_color=TEXT,
        margin=dict(l=100, r=20, t=48, b=40),
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_xaxes(gridcolor="#30363d")
    return fig


def _signals_block(signals: list) -> html.Div:
    if not signals:
        return html.Div("No rebalance signals.", style={"color": MUTED, "fontSize": "13px"})
    rows = []
    for s in signals[:12]:
        rows.append(
            html.Div(
                style={"padding": "8px", "marginBottom": "6px", "backgroundColor": "#21262d", "borderRadius": "6px"},
                children=[
                    html.Div(
                        f"{s.get('type', '?')} · pri {s.get('priority', '')}",
                        style={"fontWeight": "600", "fontSize": "13px"},
                    ),
                    html.Div(str(s.get("message", "")), style={"fontSize": "12px", "color": TEXT}),
                    html.Div(str(s.get("detail", "")), style={"fontSize": "11px", "color": MUTED}),
                ],
            )
        )
    return html.Div(
        [
            html.H4("📊 Alpha / rebalance signals", style={"color": MUTED, "marginBottom": "8px", "fontSize": "15px"}),
            html.Div(rows, style={"maxHeight": 280, "overflowY": "auto"}),
        ]
    )


_LADDER_SECTION_TITLE = "Strategy Comparison: Impact of Signals and Risk Controls"
_LADDER_INTRO = (
    "Results show that while standalone signals do not generate positive returns, the system provides "
    "insight into how correlation and regime-aware controls affect portfolio behaviour under weak alpha conditions."
)
_LADDER_ROW_GUIDE: list[tuple[str, str]] = [
    ("Baseline", "Passive allocation; serves as reference."),
    ("Vol targeting only", "Stabilises risk but does not by itself improve returns."),
    ("Signals only", "Weak predictive power in this setup; tends to underperform when costs and churn bite."),
    ("Correlation signal only", "Affects risk behaviour (tilts, conditioning) rather than standalone return generation."),
    (
        "Full system",
        "Combines signals with risk-aware controls; highlights the trade-off between alpha and risk management.",
    ),
    (
        "Random",
        "Placebo (`placebo_random_signals` in CSV) — typically worst; confirms the system is not equivalent to noise.",
    ),
]
_LADDER_KEY_TAKEAWAY = (
    "Key takeaway: risk-aware decision systems cannot compensate for weak alpha; robust signal generation "
    "is critical for performance. Negative headline metrics here are research outcomes (component tests and "
    "failure modes), not a reason to hide the table."
)


def _html_table_from_df(df: pd.DataFrame, *, max_rows: int = 80) -> html.Div:
    disp = df.head(max_rows)
    headers = [html.Th(col, style={"padding": "6px", "border": "1px solid #30363d"}) for col in disp.columns]
    body_rows = []
    for _, row in disp.iterrows():
        cells = [
            html.Td(str(row[c])[:80], style={"padding": "6px", "border": "1px solid #30363d", "fontSize": "12px"})
            for c in disp.columns
        ]
        body_rows.append(html.Tr(cells))
    return html.Div(
        style={"overflowX": "auto", "marginBottom": "12px"},
        children=[
            html.Table(
                [html.Thead(html.Tr(headers)), html.Tbody(body_rows)],
                style={"borderCollapse": "collapse", "width": "100%"},
            )
        ],
    )


def _ladder_research_section(df: pd.DataFrame) -> html.Div:
    guide_items = []
    for label, expl in _LADDER_ROW_GUIDE:
        guide_items.append(
            html.Li(
                [html.Span(label, style={"fontWeight": "600", "color": ACCENT}), f" — {expl}"],
                style={"marginBottom": "8px", "fontSize": "13px", "lineHeight": "1.45"},
            )
        )
    return html.Div(
        style={"marginTop": "20px", "marginBottom": "24px"},
        children=[
            html.H3(_LADDER_SECTION_TITLE, style={"color": TEXT, "fontSize": "18px", "marginBottom": "12px"}),
            html.P(
                _LADDER_INTRO,
                style={
                    "color": MUTED,
                    "fontSize": "14px",
                    "lineHeight": "1.55",
                    "maxWidth": "920px",
                    "marginBottom": "14px",
                },
            ),
            _html_table_from_df(df),
            html.Div(
                "How to read each row",
                style={"color": MUTED, "fontSize": "11px", "textTransform": "uppercase", "margin": "16px 0 8px"},
            ),
            html.Ul(guide_items, style={"paddingLeft": "20px", "marginTop": "0", "maxWidth": "920px"}),
            html.Div(
                _LADDER_KEY_TAKEAWAY,
                style={
                    "marginTop": "18px",
                    "padding": "14px 16px",
                    "backgroundColor": "#21262d",
                    "borderLeft": f"4px solid {ACCENT}",
                    "borderRadius": "6px",
                    "fontSize": "13px",
                    "lineHeight": "1.5",
                    "maxWidth": "920px",
                    "color": TEXT,
                },
            ),
        ],
    )


def _reverse_stress_block(stress: dict) -> html.Div:
    rev = stress.get("reverse_target_15pct")
    if not isinstance(rev, dict):
        return html.Div()
    ok = rev.get("success")
    shock = rev.get("shock") or {}
    top = sorted(shock.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
    body = [html.Div(f"Reverse stress (−15% port): Mah²={rev.get('mahalanobis_sq', 0):.3f} · ok={ok}", style={"fontSize": "12px", "color": MUTED})]
    for a, dv in top:
        body.append(html.Div(f"{a}: {dv*100:.2f}%", style={"fontSize": "12px"}))
    return html.Div(
        style={"backgroundColor": PANEL, "padding": "12px", "borderRadius": "8px", "marginTop": "12px"},
        children=[html.Div("Reverse stress (largest shocks)", style={"color": MUTED, "fontSize": "11px", "marginBottom": "6px"}), *body],
    )


def _research_content_signature() -> str:
    parts: list[str] = []
    pfig = _RESEARCH_FIG
    parts.append(f"killer:{pfig.stat().st_mtime_ns}" if pfig.is_file() else "killer:0")
    for name in (
        "ladder_table.csv",
        "leadlag_summary.csv",
        "decision_breakdown.csv",
        "decision_log.csv",
        "equity_curve.csv",
    ):
        p = _RESEARCH_OUT / name
        parts.append(f"{name}:{p.stat().st_mtime_ns}" if p.is_file() else f"{name}:0")
    return "|".join(parts)


def _build_research_page() -> html.Div:
    blocks = []
    killer = _RESEARCH_FIG
    if killer.is_file():
        b64 = base64.b64encode(killer.read_bytes()).decode("ascii")
        blocks.append(
            html.Div(
                style={"marginBottom": "20px"},
                children=[
                    html.Div("Killer chart overlay", style={"color": MUTED, "marginBottom": "8px"}),
                    html.Img(src=f"data:image/png;base64,{b64}", style={"maxWidth": "100%", "borderRadius": "8px"}),
                ],
            )
        )
    csvs = [
        "ladder_table.csv",
        "leadlag_summary.csv",
        "decision_breakdown.csv",
        "decision_log.csv",
        "equity_curve.csv",
    ]
    for name in csvs:
        p = _RESEARCH_OUT / name
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if name == "ladder_table.csv":
            blocks.append(_ladder_research_section(df))
            continue
        blocks.append(html.H4(name, style={"color": ACCENT, "marginTop": "16px"}))
        blocks.append(_html_table_from_df(df))
    if len(blocks) == 0:
        blocks.append(html.P("No research outputs found. Run backtest / research scripts to populate research/outputs.", style={"color": MUTED}))
    return html.Div(blocks)


def _build_research_page_cached() -> html.Div:
    sig = _research_content_signature()
    if _research_page_cache.get("sig") == sig and _research_page_cache.get("root") is not None:
        return _research_page_cache["root"]  # type: ignore[return-value]
    root = _build_research_page()
    _research_page_cache["sig"] = sig
    _research_page_cache["root"] = root
    return root


def create_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        style={
            "backgroundColor": DARK,
            "color": TEXT,
            "fontFamily": "Segoe UI, Roboto, sans-serif",
            "padding": "16px",
            "minHeight": "100vh",
        },
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.Interval(
                id="tick",
                interval=500,
                n_intervals=0,
                disabled=False,
            ),
            html.Div(
                id="live-wrap",
                style={"display": "block", **_WRAP_TRANSITION},
                children=_live_children(),
            ),
            html.Div(
                id="research-wrap",
                style={"display": "none", **_WRAP_TRANSITION},
                children=[_nav(), html.Div(id="research-inner")],
            ),
        ],
    )

    @app.callback(
        [
            Output("live-wrap", "style"),
            Output("research-wrap", "style"),
            Output("research-inner", "children"),
            Output("tick", "disabled"),
        ],
        [Input("url", "pathname")],
    )
    def _route(pathname):
        """Toggle panes, cache research HTML, and pause the live poll on Research (stops 500ms main-thread work)."""
        if _path_is_research(pathname):
            return (
                {"display": "none", **_WRAP_TRANSITION},
                {"display": "block", **_WRAP_TRANSITION},
                _build_research_page_cached(),
                True,
            )
        return (
            {"display": "block", **_WRAP_TRANSITION},
            {"display": "none", **_WRAP_TRANSITION},
            no_update,
            False,
        )

    @app.callback(
        [
            Output("p-banner", "children"),
            Output("p-command", "children"),
            Output("p-metrics", "children"),
            Output("p-strip", "children"),
            Output("p-explain", "children"),
            Output("p-band1", "children"),
            Output("p-killer", "figure"),
            Output("p2-var", "figure"),
            Output("p3-corr", "figure"),
            Output("p4-mc", "figure"),
            Output("p-var-trend", "figure"),
            Output("p-garch", "figure"),
            Output("p-stress", "figure"),
            Output("p5-rc", "figure"),
            Output("p-weights", "figure"),
            Output("p-signals", "children"),
            Output("p-reverse", "children"),
            Output("p6-feed", "children"),
        ],
        [Input("tick", "n_intervals")],
        [State("url", "pathname")],
    )
    def refresh(_n, pathname):
        if _path_is_research(pathname):
            raise PreventUpdate
        snap = read_snapshot()
        h = snap.header or {}
        err = snap.error
        banner = (
            html.Div(
                err,
                style={
                    "backgroundColor": "#9e6a03",
                    "color": "#fff",
                    "padding": "10px 14px",
                    "borderRadius": "6px",
                    "marginBottom": "10px",
                    "fontSize": "14px",
                    "maxWidth": "960px",
                },
            )
            if err
            else None
        )
        ss = snap.system_state or {}
        regime = str(ss.get("regime") or h.get("regime") or "?")
        rcol = _regime_color(regime)
        vp = snap.var_panel or {}
        risk_limit = float(ss.get("risk_limit_var_99", 0.05))
        vb = (snap.report or {}).get("var_block") or {}
        breach = bool(vb.get("var_breach"))
        tail_m = vp.get("tail_multiplier")
        zone = vp.get("backtesting_zone") or "—"

        band = html.Div(
            style={
                "backgroundColor": PANEL,
                "padding": "14px 16px",
                "borderRadius": "8px",
                "borderLeft": f"5px solid {rcol}",
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "12px",
                "alignItems": "stretch",
            },
            children=[
                _chip("💼 Portfolio", f"${h.get('total_value', 0):,.0f}"),
                _chip(
                    "Daily ret",
                    f"{_header_daily_pct(h):.3f}%",
                ),
                _chip("⚠️ VaR 95% (MC 1d)", f"{vp.get('mc_var_95_1d', h.get('var_95_1d', 0))*100:.2f}%"),
                _chip("⚠️ VaR 99% (MC 1d)", f"{vp.get('mc_var_99_1d', h.get('var_99_1d', 0))*100:.2f}%", accent=True),
                _chip("CF tail mult", f"{float(tail_m or 1):.2f}×"),
                _chip("Basel zone", str(zone)),
                _chip("Regime", regime, accent=True),
                _chip("Cycle", f"{snap.cycle_ms:.0f} ms"),
                _chip("VAR BREACH", "YES" if breach else "no", accent=breach),
                html.Div(h.get("timestamp", ""), style={"color": MUTED, "fontSize": "12px", "alignSelf": "center", "marginLeft": "8px"}),
            ],
        )

        corr_d = snap.correlation or {}
        strip = _build_strip(ss, vp)
        command = _build_system_command(ss)
        metrics_inst = _build_metrics_institutional(ss, vp, corr_d)
        explain = _build_explain(ss)
        fig_kill = _fig_killer_overlay(snap.overlay_series or {})
        fig_var = _fig_var_grid(vp, risk_limit)
        fig_corr = _fig_corr(corr_d)
        fig_var_trend = _fig_var_trend(vp)
        md = snap.mc_distribution or {}
        fig_mc = _fig_mc(md.get("simulations") or [], vp)
        fig_g = _fig_garch_paths(snap.garch_vol_paths or {})
        stress = (snap.report or {}).get("stress_tests") or {}
        fig_s = _fig_stress(stress if isinstance(stress, dict) else {})
        ra = snap.risk_attribution or {}
        share = ra.get("contribution_share") or {}
        if not share:
            raw = ra.get("contributions") or {}
            pv = float(ra.get("portfolio_vol") or 0.0)
            if pv > 1e-18:
                share = {k: float(v) / pv for k, v in raw.items()}
            else:
                share = dict.fromkeys(raw, 0.0)
        budget = float(ra.get("budget", 0.05))
        names = list(share.keys())[:25]
        vals_pct = [float(share[k]) * 100.0 for k in names]
        fig5 = go.Figure(
            data=[
                go.Bar(
                    x=names,
                    y=vals_pct,
                    marker_color=[
                        "#da3633" if v > budget * 100.0 * 1.25 else "#58a6ff" for v in vals_pct
                    ],
                )
            ]
        )
        fig5.add_hline(
            y=budget * 100.0,
            line_dash="dash",
            line_color=MUTED,
            annotation_text="max risk share / name",
        )
        fig5.update_layout(
            title="🛡️ Risk share per name (% of total portfolio σ) vs budget",
            paper_bgcolor=PANEL,
            plot_bgcolor=DARK,
            font_color=TEXT,
            xaxis_tickangle=-45,
            margin=dict(l=50, r=20, t=44, b=120),
        )
        fig5.update_yaxes(title="% of σₚ", gridcolor="#30363d")
        fig_w = _fig_weights(snap.weights or {}, snap.target_weights or {})
        sigs = (snap.report or {}).get("signals") or []
        sig_block = _signals_block(sigs if isinstance(sigs, list) else [])
        rev_block = _reverse_stress_block(stress if isinstance(stress, dict) else {})

        feed = list(snap.anomalies or [])
        feed.sort(key=lambda a: (_severity_rank(a.get("severity")), -float(a.get("metric_value") or 0)))
        p6 = html.Div(
            children=[
                html.Div(
                    style={
                        "backgroundColor": PANEL,
                        "padding": "8px",
                        "marginBottom": "6px",
                        "borderRadius": "4px",
                        "borderLeft": f"4px solid {_severity_border(a.get('severity'))}",
                    },
                    children=[
                        html.Div(
                            f"{a.get('layer')} · {a.get('severity')}",
                            style={"fontWeight": "600"},
                        ),
                        html.Div(str(a.get("assets", [])), style={"fontSize": "12px", "color": MUTED}),
                        html.Div(
                            f"conj={a.get('conjunction_count')} · {a.get('recommended_action')}",
                            style={"fontSize": "11px"},
                        ),
                    ],
                )
                for a in feed[-24:]
            ]
        )
        return (
            banner if banner is not None else [],
            command,
            metrics_inst,
            strip,
            explain,
            band,
            fig_kill,
            fig_var,
            fig_corr,
            fig_mc,
            fig_var_trend,
            fig_g,
            fig_s,
            fig5,
            fig_w,
            sig_block,
            rev_block,
            p6,
        )

    return app


def _live_children() -> list:
    secondary = html.Details(
        open=False,
        style={
            "marginTop": "18px",
            "backgroundColor": PANEL,
            "borderRadius": "8px",
            "padding": "12px 14px",
            "border": "1px solid #30363d",
        },
        children=[
            html.Summary(
                "🛡️ Secondary analytics — stress, vol paths, attribution, weights, signals, anomalies (expand)",
                style={"cursor": "pointer", "fontWeight": "600", "fontSize": "14px", "color": TEXT},
            ),
            html.Div(
                style={"marginTop": "14px"},
                children=[
                    html.Div(
                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                        children=[
                            html.Div(
                                style={"flex": "2 1 400px", "minWidth": 320},
                                children=[dcc.Graph(id="p-garch", style={"height": 300})],
                            ),
                            html.Div(
                                style={"flex": "2 1 400px", "minWidth": 320},
                                children=[dcc.Graph(id="p-stress", style={"height": 300})],
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "12px", "marginTop": "12px", "flexWrap": "wrap"},
                        children=[
                            html.Div(style={"flex": "1 1 400px"}, children=[dcc.Graph(id="p5-rc", style={"height": 300})]),
                            html.Div(style={"flex": "1 1 400px"}, children=[dcc.Graph(id="p-weights", style={"height": 300})]),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "12px", "marginTop": "12px", "flexWrap": "wrap"},
                        children=[
                            html.Div(style={"flex": "1 1 380px"}, children=[html.Div(id="p-signals")]),
                            html.Div(style={"flex": "1 1 380px"}, children=[html.Div(id="p-reverse")]),
                        ],
                    ),
                    html.Div(
                        style={"marginTop": "14px"},
                        children=[
                            html.H4("⚠️ Anomaly feed (severity-sorted)", style={"color": MUTED, "fontSize": "15px"}),
                            html.Div(id="p6-feed", style={"maxHeight": 320, "overflowY": "auto"}),
                        ],
                    ),
                ],
            ),
        ],
    )
    return [
        _nav(),
        html.Div(id="p-banner"),
        html.Div(id="p-command"),
        html.Div(id="p-metrics"),
        html.Div(id="p-strip"),
        html.Div(id="p-explain"),
        html.Div(
            style={"marginBottom": "10px"},
            children=[
                html.Div(
                    "⚡ VaR / risk metrics (snapshot)",
                    style={"fontSize": "12px", "color": MUTED, "textTransform": "uppercase", "fontWeight": "600", "marginBottom": "8px"},
                ),
                html.Div(id="p-band1"),
            ],
        ),
        html.Div(
            style={"marginBottom": "14px"},
            children=[
                html.Div(
                    "📈 VaR path (time series)",
                    style={"fontSize": "12px", "color": MUTED, "textTransform": "uppercase", "fontWeight": "600", "marginBottom": "8px"},
                ),
                dcc.Graph(id="p-var-trend", style={"height": 240}),
            ],
        ),
        html.Div(
            style={"marginBottom": "14px"},
            children=[
                html.Div(
                    "📉 Structure & tail (live history)",
                    style={"fontSize": "12px", "color": MUTED, "textTransform": "uppercase", "fontWeight": "600", "marginBottom": "8px"},
                ),
                dcc.Graph(id="p-killer", style={"height": 380}),
            ],
        ),
        html.Div(
            "📊 Core analytics",
            style={"fontSize": "12px", "color": MUTED, "textTransform": "uppercase", "fontWeight": "600", "marginBottom": "8px"},
        ),
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            children=[
                html.Div(style={"flex": "2 1 420px", "minWidth": 360}, children=[dcc.Graph(id="p2-var", style={"height": 340})]),
                html.Div(style={"flex": "1 1 300px", "minWidth": 260}, children=[dcc.Graph(id="p3-corr", style={"height": 340})]),
                html.Div(style={"flex": "1 1 300px", "minWidth": 260}, children=[dcc.Graph(id="p4-mc", style={"height": 340})]),
            ],
        ),
        secondary,
    ]
