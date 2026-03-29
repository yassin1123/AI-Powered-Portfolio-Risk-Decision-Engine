"""Design tokens — Ultra Elite UI (green stable / amber transition / red stress)."""

from __future__ import annotations

# Surfaces
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
    "STRESSED": RED,
    "CRISIS": RED,
}

# Spacing (8px base)
S1, S2, S3, S4, S5 = 8, 16, 24, 32, 40
RAD_SM, RAD_MD, RAD_LG = 6, 8, 12

CORR_Z_SPIKE = 1.5


def regime_color(regime: str) -> str:
    r = (regime or "").upper()
    for k, v in REGIME_BORDER.items():
        if k in r or r in k:
            return v
    return ACCENT


def regime_fill_rgba(reg: str) -> str:
    u = (reg or "").upper()
    if "STRESS" in u or "CRISIS" in u:
        return "rgba(248,81,73,0.14)"
    if "TRANSITION" in u:
        return "rgba(210,153,34,0.14)"
    return "rgba(63,185,80,0.10)"
