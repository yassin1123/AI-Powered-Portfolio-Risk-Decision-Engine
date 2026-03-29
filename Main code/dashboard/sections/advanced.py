"""Advanced systems: change detection, research links, ablation hint (UI brief §9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from dash import dcc, html

from dashboard import styles, theme

_ROOT = Path(__file__).resolve().parents[2]


def build_change_detection(ui: dict[str, Any]) -> html.Div:
    rc = ui.get("recent_changes") or {}
    if not rc or rc.get("notes") == ["insufficient_history"]:
        return html.Div(
            "Change detection: waiting for more live bars in the ring buffer.",
            style={"fontSize": "12px", "color": theme.MUTED},
        )

    chips = []
    pairs = [
        ("corr_z Δ1", rc.get("corr_z_delta_1")),
        ("corr_z Δ5", rc.get("corr_z_delta_5")),
        ("VaR Δ1", rc.get("var_99_delta_1")),
        ("conf Δ", rc.get("confidence_delta")),
        ("risk mult Δ", rc.get("risk_multiplier_delta")),
    ]
    for lab, val in pairs:
        if val is None:
            continue
        chips.append(
            html.Span(
                f"{lab}: {val}",
                style={
                    "display": "inline-block",
                    "margin": "4px 6px 4px 0",
                    "padding": "6px 10px",
                    "borderRadius": "6px",
                    "border": f"1px solid #30363d",
                    "fontSize": "11px",
                    "fontFamily": "ui-monospace, monospace",
                },
            )
        )
    if rc.get("regime_changed"):
        chips.append(
            html.Span(
                "REGIME CHANGED",
                style={
                    "display": "inline-block",
                    "margin": "4px 6px 4px 0",
                    "padding": "6px 10px",
                    "borderRadius": "6px",
                    "border": f"2px solid {theme.RED}",
                    "fontSize": "11px",
                    "fontWeight": "700",
                    "color": theme.RED,
                },
            )
        )
    notes = rc.get("notes") or []
    note_el = html.Div(" · ".join(str(n) for n in notes), style={"fontSize": "11px", "color": theme.MUTED, "marginTop": "8px"}) if notes else html.Div()

    return html.Div([html.Div(chips, style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"}), note_el])


def build_research_links_block(ui: dict[str, Any]) -> html.Div:
    rl = ui.get("research_links") or {}
    if not rl:
        return html.Div("Research links populate when elite_snapshot is present.", style={"color": theme.MUTED, "fontSize": "12px"})
    rows = []
    for k, v in sorted(rl.items()):
        rows.append(
            html.Tr(
                [
                    html.Td(str(k), style={"padding": "6px", "color": theme.MUTED, "fontSize": "12px"}),
                    html.Td(str(v), style={"padding": "6px", "color": theme.TEXT, "fontSize": "12px"}),
                ]
            )
        )
    return html.Table(html.Tbody(rows), style={"width": "100%", "borderCollapse": "collapse"})


def build_ablation_summary_strip() -> html.Div:
    p = _ROOT / "research" / "outputs" / "ablation_results.csv"
    if not p.is_file():
        return html.Div("Run scripts/run_ablations.py to create ablation_results.csv", style={"fontSize": "12px", "color": theme.MUTED})
    try:
        df = pd.read_csv(p)
    except Exception:
        return html.Div()
    if df.empty or "ablation" not in df.columns:
        return html.Div()
    row = df.iloc[0]
    cols = [c for c in df.columns if c != "ablation"]
    bits = [f"{c}={row.get(c, '')}" for c in cols[:6]]
    return html.Div(
        f"Ablation grid (first row: {row.get('ablation')}): " + " · ".join(str(b) for b in bits),
        style={"fontSize": "11px", "color": theme.MUTED, "lineHeight": "1.45"},
    )


def build_confidence_decomposition(ui: dict[str, Any]) -> html.Div:
    ss = ui.get("system_state") or {}
    risk = ui.get("risk") or {}
    tr = risk.get("decision_trace") or ss.get("decision_trace") or {}
    dec = ui.get("decision") or {}
    conf = tr.get("confidence")
    if conf is None:
        conf = (ss.get("decision") or {}).get("confidence")
    conf_f = float(conf) if conf is not None else None
    cond = dec.get("conditions_met") or tr.get("condition_flags") or {}
    items = []
    if conf_f is not None:
        items.append(html.Li(f"Trace confidence: {conf_f * 100:.0f}%", style={"marginBottom": "4px"}))
    items.extend(html.Li(f"{k}: {v}", style={"marginBottom": "4px", "fontSize": "12px", "color": theme.MUTED}) for k, v in list(cond.items())[:12])
    if not items:
        return html.Div("No decomposition available.", style={"color": theme.MUTED, "fontSize": "12px"})
    return html.Ul(items, style={"margin": "0", "paddingLeft": "18px"})


def build_advanced_section(ui: dict[str, Any]) -> html.Details:
    return html.Details(
        open=False,
        style={
            "marginTop": f"{theme.S2}px",
            "backgroundColor": theme.PANEL,
            "borderRadius": f"{theme.RAD_MD}px",
            "padding": f"{theme.S2}px",
            "border": "1px solid #30363d",
        },
        children=[
            html.Summary(
                "Advanced — change detection, scenario what-if, research hooks",
                style={"cursor": "pointer", "fontWeight": "600", "fontSize": "14px", "color": theme.TEXT},
            ),
            html.Div(
                style={"marginTop": f"{theme.S2}px"},
                children=[
                    styles.section_title("Change detection"),
                    build_change_detection(ui),
                    styles.section_title("Signal confidence & conditions"),
                    build_confidence_decomposition(ui),
                    html.P(
                        [
                            "Scenario preview (vol / corr ",
                            html.Span("z", style={"fontStyle": "italic"}),
                            ") is in the panel below — kept outside the 500ms refresh so sliders and results stay stable.",
                        ],
                        style={"fontSize": "11px", "color": theme.MUTED, "marginTop": "8px", "lineHeight": "1.45"},
                    ),
                    styles.section_title("Research & ablations"),
                    build_ablation_summary_strip(),
                    html.Div(style={"marginTop": "10px"}, children=[build_research_links_block(ui)]),
                    html.P(
                        [
                            "Full tables: click ",
                            html.Strong("RESEARCH", style={"color": theme.ACCENT}),
                            " in the top bar (walk-forward, ablations, logs).",
                        ],
                        style={"fontSize": "11px", "color": theme.MUTED, "marginTop": "10px"},
                    ),
                ],
            ),
        ],
    )


def build_scenario_panel_static() -> html.Details:
    """Static layout: must NOT live inside p-elite-stack (rebuilt every tick) or callbacks break."""
    return html.Details(
        open=True,
        className="scenario-panel-root",
        style={
            "marginTop": f"{theme.S2}px",
            "marginBottom": f"{theme.S2}px",
            "backgroundColor": theme.PANEL,
            "borderRadius": f"{theme.RAD_MD}px",
            "padding": f"{theme.S2}px",
            "border": "1px solid #30363d",
        },
        children=[
            html.Summary(
                "Scenario what-if (read-only preview — does not change the live book)",
                style={"cursor": "pointer", "fontWeight": "600", "fontSize": "14px", "color": theme.TEXT},
            ),
            html.Div(
                className="scenario-slider-wrap",
                style={
                    "marginTop": f"{theme.S2}px",
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "16px",
                    "alignItems": "flex-start",
                },
                children=[
                    html.Div(
                        [
                            html.Label("Vol ann mult", style={"fontSize": "11px", "color": theme.MUTED, "display": "block", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="scenario-vol-mult",
                                min=0.7,
                                max=1.4,
                                step=0.05,
                                value=1.0,
                                marks={0.7: "0.7×", 1.0: "1×", 1.4: "1.4×"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        style={"flex": "1 1 260px", "minWidth": 220},
                    ),
                    html.Div(
                        [
                            html.Label("Corr z add", style={"fontSize": "11px", "color": theme.MUTED, "display": "block", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="scenario-corr-add",
                                min=-1.0,
                                max=1.5,
                                step=0.1,
                                value=0.0,
                                marks={-1.0: "−1", 0.0: "0", 1.5: "+1.5"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        style={"flex": "1 1 260px", "minWidth": 220},
                    ),
                ],
            ),
            html.Div(
                [
                    html.Button(
                        "Preview scenario",
                        id="scenario-apply-btn",
                        n_clicks=0,
                        style={
                            "padding": "10px 18px",
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "cursor": "pointer",
                            "borderRadius": "6px",
                            "border": f"1px solid {theme.ACCENT}",
                            "backgroundColor": "#21262d",
                            "color": theme.TEXT,
                        },
                    ),
                    html.Span(
                        " Computes shocked market_state vs baseline (no portfolio mutation).",
                        style={"fontSize": "11px", "color": theme.MUTED, "marginLeft": "12px", "verticalAlign": "middle"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "marginTop": "12px", "marginBottom": "10px"},
            ),
            html.Div(
                id="p-scenario-out",
                style={"fontSize": "12px", "color": theme.TEXT, "minHeight": "24px"},
                children=[
                    html.Div(
                        "Set the sliders, then click “Preview scenario” to see deltas.",
                        style={"color": theme.MUTED, "fontSize": "12px"},
                    )
                ],
            ),
        ],
    )
