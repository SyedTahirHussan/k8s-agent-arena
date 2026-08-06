"""Turning runs into a table you can cite.

The reporting rule that matters: repeats are never collapsed into a boolean.
Gemini 3 is meant to run at its default temperature, so the same agent on the
same scenario genuinely differs between attempts, and "passed" would be an
anecdote dressed up as a result. Everything here is k-of-n.

Blast radius is carried as both a mean and a maximum, because one destructive run
in five is the finding, and a mean of 1.2 hides it.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from arena.runner import RunResult


@dataclass(frozen=True)
class Row:
    """Aggregated outcome for one (scenario, driver) pair."""

    scenario: str
    driver: str
    model: str
    runs: int
    passes: int
    mean_blast: float
    max_blast: int
    mean_calls: float
    total_tokens: int
    mean_seconds: float
    notes: str = ""

    @property
    def pass_rate(self) -> str:
        return f"{self.passes}/{self.runs}"


#: Stop reasons that mean the harness or the API gave out, not that the agent
#: failed the task. `budget_exhausted` is deliberately absent - running out of
#: turns is the agent failing to converge, which is a genuine result.
_INCIDENT_REASONS = {"timeout", "error", "harness_error"}


def _incidents(runs: list[RunResult]) -> str:
    """Note runs that ended for reasons that say nothing about the agent.

    Counting an API timeout as a plain failure would quietly understate a model,
    so these are surfaced next to the pass rate rather than folded into it.
    """
    reasons: dict[str, int] = {}
    for run in runs:
        if run.stop_reason in _INCIDENT_REASONS or run.error:
            key = run.stop_reason or "error"
            reasons[key] = reasons.get(key, 0) + 1

    if not reasons:
        return ""
    parts = [f"{count} run(s) {reason}" for reason, count in sorted(reasons.items())]
    return "; ".join(parts)


def aggregate(results: list[RunResult]) -> list[Row]:
    """Group runs by scenario and driver, preserving how often each passed."""
    groups: OrderedDict[tuple[str, str], list[RunResult]] = OrderedDict()
    for result in results:
        groups.setdefault((result.scenario, result.driver), []).append(result)

    rows = []
    for (scenario, driver), runs in groups.items():
        count = len(runs)
        incidents = _incidents(runs)
        rows.append(
            Row(
                scenario=scenario,
                driver=driver,
                model=runs[0].model,
                runs=count,
                passes=sum(1 for r in runs if r.passed),
                mean_blast=round(sum(r.blast_score for r in runs) / count, 2),
                max_blast=max(r.blast_score for r in runs),
                mean_calls=round(sum(r.tool_calls for r in runs) / count, 1),
                total_tokens=sum(r.total_tokens for r in runs),
                mean_seconds=round(sum(r.duration_seconds for r in runs) / count, 1),
                notes=incidents,
            )
        )
    return rows


def to_markdown(rows: list[Row]) -> str:
    """Render aggregated rows as a Markdown table."""
    if not rows:
        return "_No runs recorded._"

    header = (
        "| Scenario | Driver | Passed | Blast (mean/max) | Tool calls | Tokens | Time |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for row in rows:
        note = f" {row.notes}" if row.notes else ""
        lines.append(
            f"| {row.scenario} | {row.driver} | {row.pass_rate}{note} | "
            f"{row.mean_blast}/{row.max_blast} | {row.mean_calls} | "
            f"{row.total_tokens:,} | {row.mean_seconds}s |"
        )
    return "\n".join(lines)


def write_results(results: list[RunResult], path: Path) -> Path:
    """Persist raw run records as JSON, one object per run.

    The aggregate table is a summary; this is the evidence behind it, including
    the exact model id and timestamp each row came from.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.as_dict() for r in results], indent=2))
    return path
