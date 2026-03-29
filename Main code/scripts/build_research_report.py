"""Minimal PDF shell from latest JSON-shaped payload (extend with real stats)."""

from __future__ import annotations

import json
from pathlib import Path

from reports.generator import build_pdf_report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ladder_path = root / "research" / "outputs" / "ladder_table.csv"
    ladder_preview: list[dict[str, str]] = []
    if ladder_path.is_file():
        import csv

        with open(ladder_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                if i >= 12:
                    break
                ladder_preview.append({k: str(v) for k, v in row.items()})
    payload = {
        "report_id": "research-export",
        "generated_at": "",
        "note": "Populated from ladder_table.csv when present; extend with key_findings.md",
        "ladder_preview": ladder_preview,
    }
    out = root / "docs" / "research_report.pdf"
    build_pdf_report(payload, str(out))
    meta = root / "research" / "outputs" / "report_meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
