"""Insert or replace AUTO-GENERATED block in research/key_findings.md."""

from __future__ import annotations

import csv
from pathlib import Path


MARK_START = "<!-- AUTO-GENERATED START -->"
MARK_END = "<!-- AUTO-GENERATED END -->"


def _read_ladder(path: Path) -> str:
    if not path.is_file():
        return "_Ladder table not found; run `python -m backtest.run`._\n"
    lines = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lines.append(
                f"- **{row.get('Strategy', '')}:** Sharpe **{row.get('Sharpe', '')}**, "
                f"max DD **{row.get('max_dd', '')}**, mean turnover **{row.get('mean_turnover', '')}**"
            )
    return "\n".join(lines) + "\n" if lines else "_Empty ladder._\n"


def _read_leadlag(path: Path) -> str:
    if not path.is_file():
        return "_Lead–lag summary not found._\n"
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bullets = []
    for row in rows:
        cond = row.get("condition", "")
        n = row.get("n", "")
        dd = row.get("avg_fwd_dd", "")
        vol = row.get("avg_fwd_vol_ann", "")
        br = row.get("var_breach_rate_fwd", "")
        bullets.append(
            f"- **{cond}** (n={n}): avg forward 5d max drawdown **{dd}**, "
            f"avg ann. vol proxy **{vol}**, day breach rate **{br}**"
        )
    return "\n".join(bullets) + "\n" if bullets else "_No lead–lag rows._\n"


def _read_breakdown(path: Path) -> str:
    if not path.is_file():
        return "_Decision breakdown not found._\n"
    lines = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lines.append(
                f"- **{row.get('decision_priority', '')}:** **{row.get('pct', '')}%** "
                f"({row.get('count', '')} bars)"
            )
    return "\n".join(lines) + "\n" if lines else "_Empty breakdown._\n"


def patch_key_findings(
    key_findings_path: Path,
    ladder_csv: Path,
    leadlag_csv: Path,
    breakdown_csv: Path,
) -> None:
    block = "\n".join(
        [
            MARK_START,
            "",
            "### Quantitative snapshot (auto-generated)",
            "",
            "**Five-strategy ladder (synthetic sample):**",
            "",
            _read_ladder(ladder_csv),
            "**Lead–lag (corr_z vs forward 5-bar outcomes):**",
            "",
            _read_leadlag(leadlag_csv),
            "**Decision mix (`decision_priority`):**",
            "",
            _read_breakdown(breakdown_csv),
            "Figure: `research/figures/killer_overlay.png` (after matplotlib install and backtest run).",
            "",
            MARK_END,
            "",
        ]
    )

    text = key_findings_path.read_text(encoding="utf-8") if key_findings_path.is_file() else ""
    if MARK_START in text and MARK_END in text:
        pre, rest = text.split(MARK_START, 1)
        _, post = rest.split(MARK_END, 1)
        new_text = pre.rstrip() + "\n\n" + block + post.lstrip()
    else:
        new_text = text.rstrip() + "\n\n" + block

    key_findings_path.parent.mkdir(parents=True, exist_ok=True)
    key_findings_path.write_text(new_text, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    patch_key_findings(
        root / "research" / "key_findings.md",
        root / "research" / "outputs" / "ladder_table.csv",
        root / "research" / "outputs" / "leadlag_summary.csv",
        root / "research" / "outputs" / "decision_breakdown.csv",
    )
    print("Updated research/key_findings.md")


if __name__ == "__main__":
    main()
