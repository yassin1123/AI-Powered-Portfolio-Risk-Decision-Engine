"""Decision trace as pipeline + legacy explainer (UI brief §7)."""

from __future__ import annotations

from typing import Any

from dash import html

from dashboard import styles, theme
from dashboard.text_blocks import decision_explanation_text


def _step(title: str, body: Any, *, accent: str | None = None) -> html.Div:
    border = accent or theme.MUTED
    inner: list = [
        html.Div(
            title,
            style={
                "fontSize": "10px",
                "color": theme.MUTED,
                "textTransform": "uppercase",
                "letterSpacing": "0.08em",
                "marginBottom": "6px",
            },
        ),
    ]
    if isinstance(body, str):
        inner.append(html.Div(body, style={"fontSize": "13px", "color": theme.TEXT, "lineHeight": "1.45"}))
    else:
        inner.append(body)

    return html.Div(
        style={
            "padding": "12px 14px",
            "borderLeft": f"4px solid {border}",
            "backgroundColor": theme.DARK,
            "marginBottom": "8px",
            "borderRadius": f"{theme.RAD_SM}px",
        },
        children=inner,
    )


def _top_signals_table(title: str, d: dict[str, Any], *, limit: int = 12) -> html.Div:
    if not d:
        return html.Div("—", style={"fontSize": "12px", "color": theme.MUTED})
    items = sorted(d.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:limit]
    rows = [
        html.Tr(
            [
                html.Td(str(k), style={"padding": "4px 8px", "color": theme.MUTED, "fontSize": "11px"}),
                html.Td(f"{float(v):.4f}", style={"padding": "4px 8px", "fontFamily": "ui-monospace, monospace", "fontSize": "11px"}),
            ]
        )
        for k, v in items
    ]
    return html.Div(
        [
            html.Div(title, style={"fontSize": "11px", "color": theme.ACCENT, "marginBottom": "6px"}),
            html.Table(html.Tbody(rows), style={"borderCollapse": "collapse"}),
        ]
    )


def build_decision_trace_panel(ui: dict[str, Any]) -> html.Div:
    ss = ui.get("system_state") or {}
    dec = ui.get("decision") or {}
    risk = ui.get("risk") or {}
    tr = risk.get("decision_trace") or ss.get("decision_trace") or {}

    pre = dec.get("pre_filter_signals_top") or {}
    gated = dec.get("post_gate_signals_top") or {}
    post = dec.get("post_decision_signals_top") or {}
    cond = dec.get("conditions_met") or tr.get("condition_flags") or {}
    winning = str(dec.get("winning_rule") or tr.get("winning_rule_id") or "—")
    mult = float(dec.get("risk_multiplier") or dec.get("exposure_scale") or 1.0)
    label = str(dec.get("decision_label") or (ss.get("decision") or {}).get("priority") or "normal")

    conf = tr.get("confidence")
    if conf is None:
        conf = (ss.get("decision") or {}).get("confidence")
    conf_s = f"{float(conf) * 100:.0f}%" if conf is not None else "—"

    tree = html.Div(
        [
            _step(
                "1 · Combined signal inputs (top magnitudes)",
                _top_signals_table("Pre-filter / combined", pre if pre else post),
                accent=theme.BLUE,
            ),
            _step(
                "2 · Gates (regime / anomaly)",
                _top_signals_table("Post-gate", gated)
                if gated
                else html.Div(
                    "Post-gate snapshot not populated separately.",
                    style={"color": theme.MUTED, "fontSize": "12px"},
                ),
                accent=theme.AMBER,
            ),
            _step(
                "3 · Post-decision (engine)",
                _top_signals_table("After decision engine", post if post else pre),
                accent=theme.GREEN,
            ),
            _step(
                "4 · Rule triggers",
                html.Div(
                    [
                        html.Div(f"Winning rule: {winning}", style={"fontSize": "13px", "marginBottom": "6px"}),
                        html.Div(f"Confidence: {conf_s}", style={"fontSize": "13px", "marginBottom": "6px"}),
                        html.Div(
                            [html.Div(f"{k}: {v}", style={"fontSize": "11px", "color": theme.MUTED}) for k, v in list(cond.items())[:20]],
                        ),
                    ]
                ),
                accent=theme.ACCENT,
            ),
            _step(
                "5 · Final action",
                html.Div(
                    [
                        html.Div(f"Priority / label: {label}", style={"fontWeight": "700"}),
                        html.Div(f"Risk multiplier: {mult:.2f}×", style={"marginTop": "6px"}),
                    ]
                ),
                accent=theme.regime_color(str((ui.get("market_state") or {}).get("regime") or "")),
            ),
        ]
    )

    fq = ss.get("five_questions") or {}
    opx = str(ss.get("operator_explainer") or "")
    fq_children: list = []

    if fq:
        fq_order = [
            ("q1_state", "1. What state are we in?"),
            ("q2_why", "2. Why does the model think so?"),
            ("q3_risk_now", "3. What risk is live right now?"),
            ("q4_action", "4. What action is the engine taking?"),
            ("q5_similar_history", "5. How did similar historical states behave?"),
        ]
        fq_children.append(styles.section_title("Five questions (operator view)"))
        fq_children.append(
            html.Ul(
                [html.Li(f"{title} — {fq.get(key, '')}", style={"marginBottom": "6px"}) for key, title in fq_order if fq.get(key)],
                style={"marginTop": "0", "paddingLeft": "20px", "lineHeight": "1.5"},
            )
        )

    if opx:
        fq_children.append(styles.section_title("Plain-language snapshot"))
        fq_children.append(html.P(opx, style={"fontSize": "14px", "lineHeight": "1.55", "margin": "0"}))

    mech_lines = tr.get("mechanical_lines") or []
    drv_lines = tr.get("driver_lines") or []
    pos_lines = (tr.get("positioning") or {}).get("lines") or []
    conclusion = str(tr.get("conclusion") or "")
    policy_note = str(tr.get("policy_note") or "")

    audit_blocks: list = []
    if mech_lines or drv_lines or pos_lines or conclusion:
        audit_blocks.append(styles.section_title("Trace detail (mechanical)"))
        if mech_lines:
            audit_blocks.append(html.Ul([html.Li(x, style={"marginBottom": "4px"}) for x in mech_lines], style={"paddingLeft": "20px"}))
        if drv_lines:
            audit_blocks.append(html.Div("Score mass (audit)", style={"fontSize": "10px", "color": theme.MUTED, "marginTop": "10px"}))
            audit_blocks.append(
                html.Ul([html.Li(x, style={"marginBottom": "4px", "fontSize": "13px", "color": theme.MUTED}) for x in drv_lines], style={"paddingLeft": "20px"})
            )
        if pos_lines:
            audit_blocks.append(html.Div("Positioning", style={"fontSize": "10px", "color": theme.MUTED, "marginTop": "10px"}))
            audit_blocks.append(html.Ul([html.Li(p, style={"marginBottom": "4px"}) for p in pos_lines], style={"paddingLeft": "20px"}))
        if conclusion:
            audit_blocks.append(
                html.Div(
                    conclusion,
                    style={
                        "marginTop": "12px",
                        "padding": "12px",
                        "backgroundColor": "#21262d",
                        "borderRadius": "6px",
                        "fontWeight": "600",
                        "fontSize": "14px",
                    },
                )
            )
        if policy_note or cond:
            audit_blocks.append(
                html.Details(
                    style={"marginTop": "12px"},
                    children=[
                        html.Summary("Policy note & raw flags", style={"cursor": "pointer", "color": theme.MUTED, "fontSize": "12px"}),
                        html.Div(
                            [html.P(policy_note, style={"fontSize": "12px", "marginTop": "8px"})] if policy_note else [],
                            style={"marginTop": "6px"},
                        ),
                    ],
                )
            )

    legacy_para = decision_explanation_text(ss)
    legacy_block = html.Div(
        [
            styles.section_title("Legacy explanation (no elite trace)"),
            html.P(legacy_para, style={"fontSize": "14px", "lineHeight": "1.55", "margin": "0", "color": theme.MUTED}),
        ],
        style={"marginTop": "16px"},
    )

    show_legacy = not (pre or gated or post) and not mech_lines

    return html.Div(
        style={**styles.panel(border_left=theme.ACCENT)},
        children=[
            styles.section_title("Decision trace"),
            tree,
            *fq_children,
            *audit_blocks,
            legacy_block if show_legacy else html.Div(),
        ],
    )
