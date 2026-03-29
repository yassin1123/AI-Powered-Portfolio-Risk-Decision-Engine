"""Reusable inline style helpers for Dash components."""

from __future__ import annotations

from typing import Any

from dash import html

from dashboard import theme


def section_title(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "fontSize": "11px",
            "color": theme.MUTED,
            "textTransform": "uppercase",
            "fontWeight": "600",
            "letterSpacing": "0.06em",
            "marginBottom": f"{theme.S1}px",
        },
    )


def panel(*, border_left: str | None = None, padding: int = theme.S2) -> dict[str, Any]:
    s: dict[str, Any] = {
        "backgroundColor": theme.PANEL,
        "borderRadius": f"{theme.RAD_MD}px",
        "padding": f"{padding}px",
        "marginBottom": f"{theme.S2}px",
    }
    if border_left:
        s["borderLeft"] = f"4px solid {border_left}"
    return s


def regime_badge_style(regime: str) -> dict[str, Any]:
    c = theme.regime_color(regime)
    return {
        "display": "inline-block",
        "padding": "10px 18px",
        "borderRadius": f"{theme.RAD_LG}px",
        "border": f"3px solid {c}",
        "backgroundColor": theme.regime_fill_rgba(regime),
        "color": theme.TEXT,
        "fontSize": "clamp(20px, 3.5vw, 28px)",
        "fontWeight": "800",
        "letterSpacing": "0.04em",
    }
