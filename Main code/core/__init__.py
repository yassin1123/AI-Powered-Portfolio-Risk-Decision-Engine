"""Shared core types (snapshot boundary for Dash)."""

from core.publish import publish_snapshot, read_snapshot
from core.snapshot import DashboardSnapshot

__all__ = ["DashboardSnapshot", "publish_snapshot", "read_snapshot"]
