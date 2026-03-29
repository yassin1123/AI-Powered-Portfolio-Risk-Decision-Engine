"""Tail multiplier and reverse-stress proximity → extra hedge fraction."""

from __future__ import annotations


def tail_hedge_fraction(tail_multiplier: float, threshold: float = 1.4) -> float:
    if tail_multiplier <= threshold:
        return 0.0
    return min(0.25, (tail_multiplier - threshold) * 0.2)
