"""
Portfolio Risk Engine entrypoint: async risk loop (sole writer of DashboardSnapshot)
and Dash server (read-only callbacks). Run from project root:
  python main.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.publish import publish_snapshot
from core.snapshot import DashboardSnapshot
from dashboard.app import create_app
from data.fetcher import DataFetcher
from data.universe import get_tickers
from pre.logging_setup import configure_logging
from pre.pipeline import PipelineState, run_risk_cycle
from pre.settings import load_settings


def main() -> None:
    settings = load_settings(ROOT / "config.yaml")
    configure_logging(settings.logging)

    tickers = get_tickers(settings.universe_profile)
    fetcher = DataFetcher(settings, tickers)
    publish_snapshot(DashboardSnapshot.empty("Loading market data…"))
    fetcher.download_history()

    state = PipelineState()
    app = create_app()

    def run_dash() -> None:
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        app.run(host=settings.dash_host, port=settings.dash_port, debug=False)

    threading.Thread(target=run_dash, daemon=True).start()

    dash_url = f"http://{settings.dash_host}:{settings.dash_port}/"
    print(
        f"\n=== Portfolio Risk Engine ===\n"
        f"Dashboard: {dash_url}\n"
        "The risk loop runs until you press Ctrl+C (no single 'done' line).\n"
        "The UI refreshes about every 0.5s — that is normal Dash polling.\n"
        "Routine HTTP lines are hidden; real problems show as WARNING or ERROR.\n",
        file=sys.stderr,
        flush=True,
    )

    async def pipeline_loop() -> None:
        while True:
            if settings.simulation_mode:
                fetcher.simulation_step()
            else:
                await fetcher.fetch_batch_async()
            snap = run_risk_cycle(settings, fetcher, state)
            publish_snapshot(snap)
            delay = (
                settings.poll_interval_sim_sec
                if settings.simulation_mode
                else settings.poll_interval_live_sec
            )
            await asyncio.sleep(delay)

    try:
        asyncio.run(pipeline_loop())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
