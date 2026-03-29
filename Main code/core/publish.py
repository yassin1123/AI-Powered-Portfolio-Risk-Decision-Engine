"""Thread-safe snapshot publish/read for Dash (brief Decision record §3)."""

from __future__ import annotations

import threading

from core.snapshot import DashboardSnapshot

_lock = threading.Lock()
_snapshot: DashboardSnapshot = DashboardSnapshot.empty()


def publish_snapshot(s: DashboardSnapshot) -> None:
    global _snapshot
    with _lock:
        _snapshot = s


def read_snapshot() -> DashboardSnapshot:
    with _lock:
        return _snapshot
