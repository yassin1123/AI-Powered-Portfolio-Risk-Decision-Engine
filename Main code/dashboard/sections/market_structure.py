"""Market structure + narrative why-lines (UI brief §4–5)."""

from __future__ import annotations

from typing import Any

from dash import html

from dashboard import styles, theme


def _arrow(delta: float | None) -> str:
    if delta is None:
        return "—"
    if delta > 0.05:
        return "↑"
    if delta < -0.05:
        return "↓"
    return "→"


def build_narrative_why_block(ui: dict[str, Any]) -> html.Div:
    nar = ui.get("narrative") or {}
    lines = nar.get("why_lines") or []
    if not lines:
        return html.Div()
    items = [
        html.Li(line, style={"marginBottom": "8px", "lineHeight": "1.5"}) for line in lines if str(line).strip()
    ]
    if not items:
        return html.Div()
    return html.Div(
        style={**styles.panel(border_left=theme.ACCENT)},
        children=[
            styles.section_title("Why (narrative engine)"),
            html.Ul(items, style={"margin": "0", "paddingLeft": "20px", "color": theme.TEXT, "fontSize": "14px"}),
        ],
    )


def build_market_structure_panel(ui: dict[str, Any]) -> html.Div:
    ms = ui.get("market_state") or {}
    rc = ui.get("recent_changes") or {}
    tl = ui.get("timeline") or {}
    trans = tl.get("transition_stats") or {}

    cz = float(ms.get("corr_z") or 0.0)
    d1 = rc.get("corr_z_delta_1")
    vol_f = float(ms.get("vol_ann_forecast") or 0.0)
    vol_t = float(ms.get("vol_ann_target") or 0.1)
    pct_f = vol_f * 100.0 if vol_f < 1.5 else vol_f
    pct_t = vol_t * 100.0 if vol_t < 1.5 else vol_t
    stab = float(ms.get("stability_score") or 0.0)
    anom = int(ms.get("anomaly_count") or 0)
    tf = ms.get("trigger_flags") or {}
    fired = [k for k, v in tf.items() if v]

    row = [
        _metric(
            "Correlation (z)",
            f"{cz:.2f} {_arrow(float(d1) if d1 is not None else None)}",
            theme.RED if cz > theme.CORR_Z_SPIKE else theme.AMBER if abs(cz) >= 1.0 else theme.GREEN,
            f"Δ1b corr_z: {d1}" if d1 is not None else "",
        ),
        _metric(
            "Volatility",
            f"{pct_f:.1f}% ann vs {pct_t:.1f}% target",
            theme.RED if pct_f > pct_t * 1.15 else theme.GREEN,
            "",
        ),
        _metric(
            "Regime stability",
            f"{stab:.2f} (higher = calmer transitions)",
            theme.GREEN if stab > 0.45 else theme.AMBER,
            f"Transitions (20b): {trans.get('transitions_last_n', '—')}",
        ),
        _metric(
            "Anomalies",
            str(anom),
            theme.RED if anom >= 4 else theme.AMBER if anom >= 2 else theme.GREEN,
            ("Flags: " + ", ".join(fired[:6])) if fired else "No trigger flags set.",
        ),
    ]

    return html.Div(
        style={**styles.panel(border_left=theme.BLUE)},
        children=[
            styles.section_title("Market structure"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "10px"},
                children=row,
            ),
        ],
    )


def _metric(title: str, value: str, color: str, foot: str) -> html.Div:
    return html.Div(
        style={
            "padding": "12px",
            "borderRadius": f"{theme.RAD_SM}px",
            "border": f"2px solid {color}",
            "backgroundColor": theme.DARK,
        },
        children=[
            html.Div(title, style={"fontSize": "10px", "color": theme.MUTED, "textTransform": "uppercase"}),
            html.Div(value, style={"fontSize": "15px", "fontWeight": "700", "color": color, "marginTop": "6px"}),
            html.Div(foot, style={"fontSize": "11px", "color": theme.MUTED, "marginTop": "6px"}) if foot else html.Div(),
        ],
    )
