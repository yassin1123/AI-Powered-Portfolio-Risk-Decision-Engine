"""Risk as decision language (UI brief §6)."""

from __future__ import annotations

from typing import Any

from dash import html

from dashboard import styles, theme


def build_risk_narrative_panel(ui: dict[str, Any]) -> html.Div:
    risk = ui.get("risk") or {}
    tail = risk.get("tail") or {}
    vs_target = risk.get("vs_target") or {}

    hint = str(vs_target.get("narrative_hint") or "").strip()
    dev_bps = vs_target.get("deviation_bps")
    dev_line = ""
    if dev_bps is not None:
        dev_line = f"Forecast vs target vol: {float(dev_bps):.0f} bps."

    trend = str(tail.get("var_trend_label") or "flat")
    breach = bool(tail.get("breach_today"))
    n30 = int(tail.get("breaches_30d") or 0)
    mc = float(tail.get("mc_var_99_1d") or 0.0)
    hs = float(tail.get("hs_var_99_1d") or 0.0)
    cf = float(tail.get("cf_cvar_99") or 0.0)
    tm = float(tail.get("tail_multiplier") or 1.0)

    disagree = abs(mc - hs) > max(1e-6, hs * 0.15) if hs > 1e-9 else False

    prose_parts = []
    if trend == "increasing":
        prose_parts.append("Tail risk estimates are rising; downside asymmetry may be expanding.")
    elif trend == "decreasing":
        prose_parts.append("Tail estimates are easing versus the recent path.")
    else:
        prose_parts.append("Tail estimates are steady versus the recent path.")

    if breach:
        prose_parts.append("Live return breached the estimated VaR tail today.")
    elif n30 >= 3:
        prose_parts.append(f"Frequent tail breaches in the last 30 observations ({n30}).")

    if hint:
        prose_parts.append(hint)
    if dev_line:
        prose_parts.append(dev_line)
    if disagree:
        prose_parts.append("Historical and simulation VaR disagree materially — treat tail read with caution.")

    cluster = str(tail.get("breach_cluster_note") or "").strip()
    if cluster:
        prose_parts.append(cluster)

    detail = (
        f"MC 99% 1d VaR ≈ {mc * 100:.2f}% · HS ≈ {hs * 100:.2f}% · CF CVaR 99 ≈ {cf * 100:.2f}% · "
        f"tail mult {tm:.2f}×"
    )

    return html.Div(
        style={**styles.panel(border_left=theme.RED if breach or n30 >= 3 else theme.AMBER)},
        children=[
            styles.section_title("Live risk (decision read)"),
            html.P(
                " ".join(prose_parts),
                style={"fontSize": "15px", "lineHeight": "1.55", "color": theme.TEXT, "margin": "0 0 10px 0", "maxWidth": "920px"},
            ),
            html.Div(detail, style={"fontSize": "12px", "color": theme.MUTED, "fontFamily": "ui-monospace, monospace"}),
        ],
    )
