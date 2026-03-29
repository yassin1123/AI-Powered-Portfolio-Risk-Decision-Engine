"""Minimal PDF shell from latest JSON-shaped payload (extend with real stats)."""

from __future__ import annotations

import json
from pathlib import Path

from reports.generator import build_pdf_report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = {
        "report_id": "research-export",
        "generated_at": "",
        "note": "Populate from backtest ladder and key_findings.md",
    }
    out = root / "docs" / "research_report.pdf"
    build_pdf_report(payload, str(out))
    meta = root / "research" / "outputs" / "report_meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
