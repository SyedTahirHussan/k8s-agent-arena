"""Orchestration: provision, break, hand over, grade, measure, destroy.

The ordering matters more than it looks. The baseline snapshot is taken *after*
the scenario's manifests have settled, so the churn Kubernetes generates while
creating the broken state is not billed to the agent. Grading happens before the
final snapshot, because grading polls and the cluster should be quiet by the time
its end state is recorded.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from arena.checks import ClusterView
from arena.cluster import ClusterError, apply_manifests, arena_cluster, snapshot
from arena.drivers.base import Budget, Transcript
from arena.kubectl import KubectlSurface, LiveClusterView
from arena.scenario import Scenario
from arena.settle import evaluate_until_passing
from arena.state import blast_radius, diff

#: How long to let the scenario's broken state materialise before snapshotting.
#: Too short and the setup's own churn is charged to the agent as collateral.
SETUP_SETTLE_SECONDS = 20


@dataclass(frozen=True)
class RunResult:
    """One agent's attempt at one scenario."""

    scenario: str
    title: str
    driver: str
    model: str
    passed: bool
    checks_passed: int
    checks_total: int
    verification: str
    blast_score: int
    blast_summary: str
    collateral: list[str] = field(default_factory=list)
    tool_calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    stop_reason: str = ""
    summary: str = ""
    started_at: str = ""
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return asdict(self)


def _failed_run(scenario: Scenario, driver, started_at: str, error: str) -> RunResult:
    """A run that never got as far as being graded still has to be reported."""
    return RunResult(
        scenario=scenario.id,
        title=scenario.title,
        driver=getattr(driver, "name", str(driver)),
        model=getattr(driver, "model", ""),
        passed=False,
        checks_passed=0,
        checks_total=len(scenario.verification.checks),
        verification="run did not complete",
        blast_score=0,
        blast_summary="not measured",
        stop_reason="harness_error",
        started_at=started_at,
        error=error,
    )


def attempt(driver, task: str, tools, budget: Budget) -> Transcript:
    """Run ``driver``, turning a crash into a recorded failure rather than an exit.

    A driver that raises is one failed attempt, not a failed sweep. The cluster is
    still standing, so the run keeps its grading and its blast radius and simply
    does not pass - where letting the exception out would discard every result
    collected so far, up to an hour of them.

    KeyboardInterrupt and SystemExit are deliberately not caught: someone
    interrupting a sweep wants it to stop, not to receive a report of
    interruptions.
    """
    try:
        return driver.run(task=task, tools=tools, budget=budget)
    except Exception as exc:  # noqa: BLE001 - the whole point is to not propagate
        return Transcript(
            driver=getattr(driver, "name", str(driver)),
            model=getattr(driver, "model", ""),
            stop_reason="harness_error",
            summary=f"driver raised {type(exc).__name__}: {exc}",
        )


def run_scenario(
    scenario: Scenario,
    driver,
    budget: Budget | None = None,
    keep: bool = False,
    settle_seconds: int = SETUP_SETTLE_SECONDS,
) -> RunResult:
    """Run one scenario against one driver on a throwaway cluster."""
    budget = budget or Budget()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.monotonic()

    try:
        with arena_cluster(keep=keep) as handle:
            manifests = scenario.manifests_path()
            if manifests is not None:
                apply_manifests(manifests, handle.context, scenario.namespace)

            # Let the fault materialise before we start attributing changes.
            time.sleep(settle_seconds)
            before = snapshot(handle.context)

            tools = KubectlSurface(context=handle.context)
            transcript = attempt(driver, scenario.task, tools, budget)

            view: ClusterView = LiveClusterView(context=handle.context)
            outcome = evaluate_until_passing(
                scenario.verification, view, timeout_seconds=scenario.timeout_seconds
            )

            after = snapshot(handle.context)
            radius = blast_radius(diff(before, after), scenario.in_scope)

    except ClusterError as exc:
        return _failed_run(scenario, driver, started_at, str(exc))

    collateral = sorted(
        str(ref) for ref in (radius.created | radius.deleted | radius.modified)
    )

    return RunResult(
        scenario=scenario.id,
        title=scenario.title,
        driver=transcript.driver,
        model=transcript.model,
        passed=outcome.passed,
        checks_passed=outcome.passed_count,
        checks_total=len(outcome.results),
        verification=outcome.summary(),
        blast_score=radius.score,
        blast_summary=radius.summary(),
        collateral=collateral,
        tool_calls=len(transcript.calls),
        failed_calls=transcript.failed_calls,
        input_tokens=transcript.input_tokens,
        output_tokens=transcript.output_tokens,
        duration_seconds=round(time.monotonic() - started, 1),
        stop_reason=transcript.stop_reason,
        summary=transcript.summary,
        started_at=started_at,
    )


def run_suite(
    scenarios: list[Scenario],
    driver_factory,
    repeats: int = 1,
    budget: Budget | None = None,
    on_result=None,
) -> list[RunResult]:
    """Run every scenario ``repeats`` times.

    ``driver_factory`` is called per run rather than reused, so no state leaks
    between attempts. Results are yielded to ``on_result`` as they land, because
    a sweep is slow and a crash on run 14 should not lose the first thirteen.
    """
    results: list[RunResult] = []
    for scenario in scenarios:
        for _ in range(repeats):
            result = run_scenario(scenario, driver_factory(), budget=budget)
            results.append(result)
            if on_result:
                on_result(result)
    return results
