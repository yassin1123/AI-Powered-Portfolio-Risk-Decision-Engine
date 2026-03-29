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
from dashboard.sections.advanced import build_advanced_section, build_scenario_panel_static
from dashboard.sections.decision_trace import build_decision_trace_panel
from dashboard.sections.hero import build_hero_panel
from dashboard.sections.historical import (
    build_analogs_panel,
    build_regime_performance_panel,
    build_timeline_panel,
    killer_chart_caption,
)
from dashboard.sections.market_structure import build_market_structure_panel, build_narrative_why_block
from dashboard.sections.risk_narrative import build_risk_narrative_panel
from dashboard.text_blocks import action_line
from dashboard.theme import (
    ACCENT,
    AMBER,
    BLUE,
    DARK,
    GREEN,
    MUTED,
    PANEL,
    RED,
    TEXT,
    regime_color,
    regime_fill_rgba,
)
from dashboard.view_model import build_ui_model


def _header_daily_pct(h: dict) -> float:
    """Simple return % from header (pipeline sends log-return sum + optional expm1 field)."""
    if h.get("daily_return_simple_approx") is not None:
        return float(h["daily_return_simple_approx"]) * 100.0
    return float(np.expm1(float(h.get("daily_return", 0.0)))) * 100.0


def _path_is_research(pathname: str | None) -> bool:
    if pathname is None:
        return False
    p = pathname.lower().strip().rstrip("/") or "/"
    return p == "/research"

_ROOT = Path(__file__).resolve().parent.parent
_RESEARCH_OUT = _ROOT / "research" / "outputs"
_RESEARCH_FIG = _ROOT / "research" / "figures" / "killer_overlay.png"

# Rebuild research tab only when output files change (navigation was re-reading CSVs + PNG every time).
_research_page_cache: dict[str, object | None] = {"sig": None, "root": None}

_WRAP_TRANSITION = {
    "transition": "opacity 0.18s ease-out",
    "willChange": "opacity",
}

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


def _portfolio_context_chips(corr_d: dict, vp: dict) -> html.Div:
    pct = vp.get("var_99_percentile_vs_history")
    pct_s = f"{pct:.0f}th" if pct is not None and isinstance(pct, (int, float)) else "—"
    br = int(vp.get("breaches_30d") or 0)
    trend = str(vp.get("var_trend_label") or "—")
    div_s = float(corr_d.get("diversification_score") or 0.0)
    chips = [
        _strip_chip("Diversification (1−ρ̄)", f"{div_s:.2f}", GREEN if div_s > 0.45 else AMBER),
        _strip_chip("VaR pctile", pct_s, AMBER if pct is not None and pct > 75 else GREEN),
        _strip_chip("Breaches (30d)", str(br), RED if br >= 3 else GREEN),
        _strip_chip("VaR trend", trend.upper(), RED if trend == "increasing" else GREEN),
    ]
    return html.Div(
        style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "marginBottom": "14px"},
        children=chips,
    )


def _fig_killer_overlay(ov: dict) -> go.Figure:
    fs = ov.get("full_span")
    if isinstance(fs, dict) and fs.get("dates") and len(fs["dates"]) >= 2:
        return _fig_killer_overlay_full_span(fs)
    return _fig_killer_overlay_cycle(ov)


def _fig_killer_overlay_full_span(fs: dict) -> go.Figure:
    dates = list(fs["dates"])
    dd = [float(x) for x in (fs.get("drawdown_pct") or [])]
    cz_raw = fs.get("corr_z") or []
    cz: list[float] = []
    for x in cz_raw:
        try:
            v = float(x)
            cz.append(v if np.isfinite(v) else 0.0)
        except (TypeError, ValueError):
            cz.append(0.0)
    reg = [str(x) for x in (fs.get("regime") or [])]
    if len(reg) != len(dates):
        reg = ["NORMAL"] * len(dates)
    n = min(len(dates), len(dd), len(cz), len(reg))
    if n < 2:
        return _empty_killer_fig("Computing full history overlay…")

    dates, dd, cz, reg = dates[:n], dd[:n], cz[:n], reg[:n]
    max_pts = 3500
    if n > max_pts:
        step = max(1, n // max_pts)
        dates = dates[::step]
        dd = dd[::step]
        cz = cz[::step]
        reg = reg[::step]
        n = len(dates)

    x_dt = pd.to_datetime(dates)
    h0 = str(fs.get("history_start") or "")
    h1 = str(fs.get("history_end") or "")
    nb = fs.get("n_bars", n)
    title = (
        f"📉 Drawdown vs correlation Z — {h0} → {h1}<br>"
        f"<sup>{nb} trading days (downsampled for display if very long)</sup>"
    )

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
                x0=x_dt[i0],
                x1=x_dt[i1],
                y0=0,
                y1=1,
                fillcolor=regime_fill_rgba(reg[i0]),
                line_width=0,
                layer="below",
            )
        )
        i0 = i if i < n else i0

    fig.update_layout(shapes=shapes)
    fig.add_trace(
        go.Scatter(
            x=x_dt,
            y=dd,
            mode="lines",
            name="Portfolio drawdown %",
            line=dict(color="#58a6ff", width=1.5),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x_dt,
            y=cz,
            mode="lines",
            name="Correlation instability (z)",
            line=dict(color="#ffa657", width=1.5),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        paper_bgcolor=PANEL,
        plot_bgcolor=DARK,
        font_color=TEXT,
        hovermode="x unified",
        margin=dict(t=72, b=48, l=58, r=58),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor="#30363d", title="Date")
    fig.update_yaxes(title_text="Drawdown %", secondary_y=False, gridcolor="#30363d")
    fig.update_yaxes(title_text="Z-score", secondary_y=True, gridcolor="#30363d")
    return fig


def _empty_killer_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
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


def _fig_killer_overlay_cycle(ov: dict) -> go.Figure:
    dd = ov.get("drawdown") or []
    cz = ov.get("corr_z") or []
    reg = ov.get("regime") or []
    n = min(len(dd), len(cz), len(reg), 5000)
    if n < 2:
        return _empty_killer_fig("Live history builds as the risk loop runs…")

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
                fillcolor=regime_fill_rgba(reg[i0]),
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
            "gap": "20px",
            "alignItems": "center",
            "flexWrap": "wrap",
        },
        children=[
            html.Span("Risk cockpit", style={"fontWeight": "700", "fontSize": "18px", "marginRight": "12px"}),
            dcc.Link(
                "LIVE",
                href="/",
                refresh=False,
                style={"color": ACCENT, "textDecoration": "none", "fontWeight": "600", "padding": "6px 10px", "borderRadius": "6px"},
            ),
            dcc.Link(
                "RESEARCH",
                href="/research",
                refresh=False,
                style={
                    "color": TEXT,
                    "textDecoration": "none",
                    "fontWeight": "600",
                    "padding": "6px 10px",
                    "borderRadius": "6px",
                    "border": "1px solid #30363d",
                    "backgroundColor": "#21262d",
                },
            ),
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
        "equity_curve_gross.csv",
        "hero_signal_validation_buckets.csv",
        "ablation_results.csv",
        "cost_sensitivity.csv",
    ):
        p = _RESEARCH_OUT / name
        parts.append(f"{name}:{p.stat().st_mtime_ns}" if p.is_file() else f"{name}:0")
    return "|".join(parts)


def _build_research_page() -> html.Div:
    blocks = [
        html.Div(
            style={
                "marginBottom": "16px",
                "padding": "12px 14px",
                "backgroundColor": PANEL,
                "borderRadius": "8px",
                "borderLeft": f"4px solid {ACCENT}",
            },
            children=[
                html.Div("Research spine", style={"fontSize": "11px", "color": MUTED, "textTransform": "uppercase", "marginBottom": "6px"}),
                html.Ul(
                    [
                        html.Li([html.Code("python -m backtest.walkforward"), " — walk-forward OOS"]),
                        html.Li([html.Code("python scripts/run_ablations.py"), " — component ablations"]),
                        html.Li([html.Code("python -m backtest.run"), " — equity + decision_log export"]),
                    ],
                    style={"margin": "0", "paddingLeft": "20px", "fontSize": "13px", "lineHeight": "1.6"},
                ),
            ],
        )
    ]
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
        "equity_curve_gross.csv",
        "hero_signal_validation_buckets.csv",
        "hero_signal_validation_series.csv",
        "ablation_results.csv",
        "cost_sensitivity.csv",
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
    _dash_dir = Path(__file__).resolve().parent
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder=str(_dash_dir / "assets"),
    )
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
            html.Div(
                style={
                    "position": "sticky",
                    "top": "0",
                    "zIndex": "2000",
                    "backgroundColor": DARK,
                    "paddingBottom": "10px",
                    "marginBottom": "8px",
                    "borderBottom": "1px solid #30363d",
                },
                children=[_nav()],
            ),
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
                children=[html.Div(id="research-inner")],
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
            Output("p-elite-stack", "children"),
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
            Output("prev-cycle-meta", "data"),
        ],
        [Input("tick", "n_intervals")],
        [State("url", "pathname"), State("prev-cycle-meta", "data")],
    )
    def refresh(_n, pathname, prev_meta):
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
        ui = build_ui_model(snap)
        regime = str(ss.get("regime") or h.get("regime") or ui.get("regime") or "?")
        rcol = regime_color(regime)
        vp = snap.var_panel or {}
        risk_limit = float(ss.get("risk_limit_var_99", 0.05))
        vb = (snap.report or {}).get("var_block") or {}
        breach = bool(vb.get("var_breach"))
        tail_m = vp.get("tail_multiplier")
        zone = vp.get("backtesting_zone") or "—"

        new_meta = {
            "regime": str(ui.get("regime") or regime),
            "headline": str((ui.get("narrative") or {}).get("headline") or ""),
            "cycle": int((ui.get("meta") or {}).get("cycle") or 0),
        }
        flash = bool(
            prev_meta
            and (
                prev_meta.get("regime") != new_meta.get("regime")
                or prev_meta.get("headline") != new_meta.get("headline")
            )
        )

        elite_stack = html.Div(
            [
                build_hero_panel(ui, flash_border=flash, legacy_action_line=action_line(ss)),
                build_narrative_why_block(ui),
                build_market_structure_panel(ui),
                _portfolio_context_chips(snap.correlation or {}, vp),
                build_risk_narrative_panel(ui),
                html.Div(
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-start"},
                    children=[
                        html.Div(style={"flex": "1 1 400px", "minWidth": 320}, children=[build_timeline_panel(ui)]),
                        html.Div(style={"flex": "1 1 400px", "minWidth": 320}, children=[build_analogs_panel(ui)]),
                    ],
                ),
                build_regime_performance_panel(ui),
                build_decision_trace_panel(ui),
                build_advanced_section(ui),
            ]
        )

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
            elite_stack,
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
            new_meta,
        )

    @app.callback(
        Output("p-scenario-out", "children"),
        Input("scenario-apply-btn", "n_clicks"),
        [
            State("scenario-vol-mult", "value"),
            State("scenario-corr-add", "value"),
            State("url", "pathname"),
        ],
        prevent_initial_call=True,
    )
    def scenario_whatif(n_clicks, vol_m, corr_add, pathname):
        if _path_is_research(pathname):
            raise PreventUpdate
        if not n_clicks:
            raise PreventUpdate
        snap = read_snapshot()
        elite = (snap.report or {}).get("elite_snapshot")
        if not isinstance(elite, dict) or not elite.get("market_state"):
            return html.Div("Elite snapshot not available — run live pipeline with elite_snapshot.", style={"color": MUTED})
        from scenario.shocks import delta_vs_base, shock_market_state

        ms = elite["market_state"]
        vm = float(vol_m) if vol_m is not None else 1.0
        ca = float(corr_add) if corr_add is not None else 0.0
        shocked, dvec = shock_market_state(ms, vol_ann_mult=vm, corr_z_add=ca)
        diff = delta_vs_base(ms, shocked)
        lines = [
            html.Div("Shocked market_state deltas (read-only)", style={"fontWeight": "600", "marginBottom": "6px"}),
            html.Pre(
                str(dvec)[:1200],
                style={"fontSize": "11px", "color": MUTED, "whiteSpace": "pre-wrap", "margin": "0"},
            ),
        ]
        if diff:
            lines.append(html.Div("Recursive diff keys", style={"marginTop": "8px", "fontSize": "11px", "color": ACCENT}))
            lines.append(html.Pre(str(diff)[:800], style={"fontSize": "10px", "color": TEXT, "whiteSpace": "pre-wrap"}))
        return html.Div(lines)

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
        dcc.Store(id="prev-cycle-meta", data=None),
        html.Div(id="p-banner"),
        html.Div(id="p-elite-stack"),
        build_scenario_panel_static(),
        html.Div(
            style={"marginBottom": "10px"},
            children=[
                html.Div(
                    "VaR and limits — snapshot chips (evidence)",
                    style={"fontSize": "12px", "color": MUTED, "textTransform": "uppercase", "fontWeight": "600", "marginBottom": "8px"},
                ),
                html.Div(id="p-band1"),
            ],
        ),
        html.Div(
            style={"marginBottom": "14px"},
            children=[
                html.Div(
                    "VaR path — time series",
                    style={"fontSize": "12px", "color": MUTED, "textTransform": "uppercase", "fontWeight": "600", "marginBottom": "8px"},
                ),
                dcc.Graph(id="p-var-trend", style={"height": 240}),
            ],
        ),
        html.Div(
            style={"marginBottom": "14px"},
            children=[
                html.Div(
                    "Historical structure — drawdown vs correlation stress",
                    style={"fontSize": "12px", "color": MUTED, "textTransform": "uppercase", "fontWeight": "600", "marginBottom": "8px"},
                ),
                killer_chart_caption(),
                dcc.Graph(id="p-killer", style={"height": 380}),
            ],
        ),
        html.Div(
            "Core analytics — distribution and correlation",
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
