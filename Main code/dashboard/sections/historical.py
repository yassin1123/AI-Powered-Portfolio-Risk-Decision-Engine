"""Historical context: timeline, analogs, regime attribution (UI brief §8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from dash import html

from dashboard import styles, theme

_ROOT = Path(__file__).resolve().parents[2]


def killer_chart_caption() -> html.Div:
    return html.Div(
        "Full downloaded history (from config history_start through the latest bar): portfolio drawdown vs correlation z, shaded by stress bucket — structural context only.",
        style={"fontSize": "12px", "color": theme.MUTED, "lineHeight": "1.45", "marginBottom": "8px", "maxWidth": "900px"},
    )


def build_timeline_panel(ui: dict[str, Any]) -> html.Div:
    tl = ui.get("timeline") or {}
    segments = tl.get("segments") or []
    trans = tl.get("transition_stats") or {}
    if not segments:
        return html.Div(
            style={**styles.panel(border_left=theme.MUTED)},
            children=[
                styles.section_title("Regime timeline"),
                html.Div("Building history as the live loop runs…", style={"color": theme.MUTED, "fontSize": "13px"}),
            ],
        )

    rows = []
    for seg in segments[-12:]:
        rows.append(
            html.Tr(
                [
                    html.Td(str(seg.get("regime", "")), style=_td()),
                    html.Td(str(seg.get("start_timestamp", ""))[:19], style=_td()),
                    html.Td(str(seg.get("end_timestamp", ""))[:19], style=_td()),
                    html.Td(str(seg.get("bars", "")), style=_td()),
                    html.Td(str(seg.get("mean_corr_z", "—")), style=_td()),
                    html.Td(str(seg.get("mean_confidence", "—")), style=_td()),
                ]
            )
        )
    head = html.Tr(
        [
            html.Th("Regime", style=_th()),
            html.Th("Start", style=_th()),
            html.Th("End", style=_th()),
            html.Th("Bars", style=_th()),
            html.Th("Mean z", style=_th()),
            html.Th("Mean conf", style=_th()),
        ]
    )
    hint = ""
    if trans:
        hint = f"Transitions (last window): {trans.get('transitions_last_n', '—')} · persistence: {trans.get('persistence_hint', '—')}"

    return html.Div(
        style={**styles.panel(border_left=theme.GREEN)},
        children=[
            styles.section_title("Regime timeline (recent segments)"),
            html.Table([html.Thead(head), html.Tbody(rows)], style={"borderCollapse": "collapse", "width": "100%", "fontSize": "12px"}),
            html.Div(hint, style={"fontSize": "11px", "color": theme.MUTED, "marginTop": "8px"}) if hint else html.Div(),
        ],
    )


def _th() -> dict[str, Any]:
    return {"textAlign": "left", "padding": "6px", "borderBottom": f"1px solid #30363d", "color": theme.MUTED}


def _td() -> dict[str, Any]:
    return {"padding": "6px", "borderBottom": f"1px solid #21262d", "color": theme.TEXT}


def build_analogs_panel(ui: dict[str, Any]) -> html.Div:
    an = ui.get("analogs") or {}
    neighbors = an.get("neighbors") or []
    note = str(an.get("note") or "")
    if not neighbors:
        return html.Div(
            style={**styles.panel(border_left=theme.MUTED)},
            children=[
                styles.section_title("Similar past states"),
                html.Div(note or "Insufficient live history for kNN neighbors yet.", style={"color": theme.MUTED, "fontSize": "13px"}),
            ],
        )

    rows = []
    for n in neighbors:
        feat = n.get("features") or {}
        fv = ", ".join(f"{k}={v:.3f}" for k, v in list(feat.items())[:5])
        rows.append(
            html.Tr(
                [
                    html.Td(str(n.get("history_index", "")), style=_td()),
                    html.Td(f"{float(n.get('distance', 0)):.4f}", style=_td()),
                    html.Td(fv, style=_td()),
                    html.Td(str(n.get("forward_5d_return", "—")), style=_td()),
                    html.Td(str(n.get("forward_10d_vol", "—")), style=_td()),
                ]
            )
        )
    head = html.Tr(
        [
            html.Th("Idx", style=_th()),
            html.Th("Distance", style=_th()),
            html.Th("Features", style=_th()),
            html.Th("Fwd 5d ret", style=_th()),
            html.Th("Fwd 10d vol", style=_th()),
        ]
    )
    return html.Div(
        style={**styles.panel(border_left=theme.ACCENT)},
        children=[
            styles.section_title("Similar past states (past-only kNN)"),
            html.Table([html.Thead(head), html.Tbody(rows)], style={"borderCollapse": "collapse", "width": "100%", "fontSize": "11px"}),
            html.Div(note, style={"fontSize": "11px", "color": theme.MUTED, "marginTop": "6px"}) if note else html.Div(),
        ],
    )


def build_regime_performance_panel(ui: dict[str, Any]) -> html.Div:
    rl = ui.get("research_links") or {}
    p = _ROOT / "research" / "outputs" / "decision_log.csv"
    if not p.is_file():
        hint = rl.get("by_regime_metrics") or "research.by_regime_metrics"
        return html.Div(
            style={**styles.panel(border_left=theme.MUTED)},
            children=[
                styles.section_title("Performance by regime"),
                html.Div(
                    [
                        html.P("No decision_log.csv in research/outputs. Run a backtest export to populate.", style={"color": theme.MUTED, "fontSize": "13px"}),
                        html.P(f"Offline helper: {hint}", style={"color": theme.MUTED, "fontSize": "11px"}),
                    ]
                ),
            ],
        )

    try:
        df = pd.read_csv(p)
        log = df.to_dict("records")
    except Exception:
        return html.Div("Could not read decision_log.csv", style={"color": theme.RED})

    try:
        from research.by_regime_metrics import performance_by_regime

        out = performance_by_regime(log)
    except Exception as e:
        return html.Div(f"Regime metrics error: {e}", style={"color": theme.RED})

    by = out.get("by_regime") or {}
    warns = out.get("warnings") or []
    blocks = []
    for reg, block in sorted(by.items()):
        m = block.get("metrics") or {}
        n = block.get("n_bars", 0)
        line = f"{reg}: n={n}"
        if m:
            line += f" · Sharpe {m.get('sharpe', 0):.2f} · maxDD {m.get('max_dd', 0):.2%} · CAGR {m.get('cagr', 0):.2%}"
        blocks.append(html.Div(line, style={"fontSize": "12px", "marginBottom": "6px", "fontFamily": "ui-monospace, monospace"}))

    return html.Div(
        style={**styles.panel(border_left=theme.BLUE)},
        children=[
            styles.section_title("Performance by regime (offline log)"),
            html.Div(blocks),
            html.Div("; ".join(warns), style={"fontSize": "11px", "color": theme.AMBER, "marginTop": "8px"}) if warns else html.Div(),
        ],
    )
