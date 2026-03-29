"""Hero decision panel (UI brief §3)."""

from __future__ import annotations

from typing import Any

from dash import html

from dashboard import styles, theme


def _tail_prose(tail: dict[str, Any]) -> str:
    if not tail:
        return "Tail risk metrics loading."
    trend = str(tail.get("var_trend_label") or "flat")
    breach = bool(tail.get("breach_today"))
    n30 = int(tail.get("breaches_30d") or 0)
    parts = []
    if trend == "increasing":
        parts.append("VaR estimate drifting up — breach risk elevated.")
    elif trend == "decreasing":
        parts.append("VaR estimate easing.")
    else:
        parts.append("VaR estimate stable versus recent path.")
    if breach:
        parts.append("Today's return breached the estimated tail.")
    elif n30 >= 3:
        parts.append(f"{n30} breaches in the last 30 bars — cluster risk.")
    note = str(tail.get("breach_cluster_note") or "").strip()
    if note:
        parts.append(note)
    return " ".join(parts)


def _vol_driver(ms: dict[str, Any], vs_target: dict[str, Any]) -> str:
    f = float(ms.get("vol_ann_forecast") or 0.0)
    t = float(ms.get("vol_ann_target") or 0.1)
    hint = str(vs_target.get("narrative_hint") or "")
    if f > t * 1.1:
        tag = "elevated vs target"
    elif f < t * 0.9:
        tag = "below target"
    else:
        tag = "near target"
    pct_f = f * 100.0 if f < 1.5 else f
    pct_t = t * 100.0 if t < 1.5 else t
    base = f"Forecast vol ~{pct_f:.1f}% ann vs target ~{pct_t:.1f}% ({tag})."
    return (base + " " + hint).strip()


def _corr_driver(ms: dict[str, Any]) -> str:
    z = float(ms.get("corr_z") or 0.0)
    b = str(ms.get("corr_bucket") or "")
    lvl = float(ms.get("corr_level") or 0.0)
    if z > theme.CORR_Z_SPIKE:
        return f"Correlation stress high (z={z:.2f}, ρ̄≈{lvl:.2f}, bucket {b})."
    if z < -0.5:
        return f"Correlation unusually low (z={z:.2f}) — diversification more available."
    return f"Correlation z={z:.2f}, ρ̄≈{lvl:.2f}, bucket {b}."


def build_hero_panel(
    ui: dict[str, Any],
    *,
    flash_border: bool = False,
    legacy_action_line: str = "",
) -> html.Div:
    ms = ui.get("market_state") or {}
    nar = ui.get("narrative") or {}
    dec = ui.get("decision") or {}
    risk = ui.get("risk") or {}
    tail = risk.get("tail") or {}
    vs_target = risk.get("vs_target") or {}
    tr = risk.get("decision_trace") or ui.get("system_state", {}).get("decision_trace") or {}

    regime = str(ms.get("regime") or ui.get("regime") or "?")
    rcol = theme.regime_color(regime)

    conf = float(
        tr.get("confidence")
        if tr.get("confidence") is not None
        else ms.get("regime_confidence")
        if ms.get("regime_confidence") is not None
        else ui.get("confidence", 0.55)
    )
    conf = max(0.0, min(1.0, conf))

    headline = str(nar.get("headline") or f"Regime {regime}")
    summary = str(nar.get("summary") or "")
    action = str(nar.get("action_line") or legacy_action_line or "").strip()
    if not action:
        mult = float(dec.get("risk_multiplier") or dec.get("exposure_scale") or 1.0)
        action = f"Scale exposure to ~{mult * 100:.0f}% of baseline."
        if dec.get("activate_hedge"):
            action += " Hedge overlay armed."

    why_preview = nar.get("why_lines") or []
    subcopy = summary or (why_preview[0] if why_preview else "")

    border = f"5px solid {('#ffa657' if flash_border else rcol)}"
    conf_bar_w = f"{conf * 100:.0f}%"

    headline_block_children: list = [
        html.Div(
            headline,
            style={
                "fontSize": "clamp(18px, 2.8vw, 24px)",
                "fontWeight": "700",
                "color": theme.TEXT,
                "lineHeight": "1.25",
            },
        ),
    ]
    if subcopy:
        headline_block_children.append(
            html.Div(
                subcopy,
                style={
                    "fontSize": "14px",
                    "color": theme.MUTED,
                    "marginTop": "8px",
                    "lineHeight": "1.5",
                    "maxWidth": "720px",
                },
            )
        )

    drivers = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
            "gap": f"{theme.S1}px",
            "marginTop": f"{theme.S2}px",
        },
        children=[
            _driver_cell("Correlation", _corr_driver(ms)),
            _driver_cell("Volatility", _vol_driver(ms, vs_target)),
            _driver_cell("Anomalies", f"Count {int(ms.get('anomaly_count') or 0)}."),
            _driver_cell("Tail / VaR", _tail_prose(tail)),
        ],
    )

    return html.Div(
        style={
            **styles.panel(border_left=None, padding=theme.S3),
            "border": border,
            "borderRadius": f"{theme.RAD_LG}px",
            "backgroundColor": theme.DARK,
        },
        children=[
            styles.section_title("Decision cockpit"),
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": f"{theme.S3}px", "alignItems": "flex-start"},
                children=[
                    html.Div(str(regime), style=styles.regime_badge_style(regime)),
                    html.Div(
                        style={"flex": "1 1 280px", "minWidth": 240},
                        children=headline_block_children,
                    ),
                    html.Div(
                        style={"minWidth": "140px"},
                        children=[
                            html.Div("Confidence", style={"fontSize": "10px", "color": theme.MUTED}),
                            html.Div(f"{conf * 100:.0f}%", style={"fontSize": "28px", "fontWeight": "800"}),
                            html.Div(
                                style={
                                    "height": "8px",
                                    "backgroundColor": theme.PANEL,
                                    "borderRadius": "4px",
                                    "marginTop": "8px",
                                    "overflow": "hidden",
                                },
                                children=[
                                    html.Div(
                                        style={
                                            "width": conf_bar_w,
                                            "height": "100%",
                                            "backgroundColor": theme.GREEN if conf > 0.55 else theme.AMBER,
                                        }
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                action,
                style={
                    "marginTop": f"{theme.S2}px",
                    "fontSize": "16px",
                    "fontWeight": "700",
                    "color": theme.AMBER,
                    "lineHeight": "1.45",
                },
            ),
            drivers,
        ],
    )


def _driver_cell(title: str, body: str) -> html.Div:
    return html.Div(
        style={
            "padding": f"{theme.S1}px {theme.S2}px",
            "backgroundColor": theme.PANEL,
            "borderRadius": f"{theme.RAD_SM}px",
            "border": f"1px solid #30363d",
        },
        children=[
            html.Div(title, style={"fontSize": "10px", "color": theme.MUTED, "textTransform": "uppercase"}),
            html.Div(body, style={"fontSize": "12px", "color": theme.TEXT, "marginTop": "6px", "lineHeight": "1.45"}),
        ],
    )
