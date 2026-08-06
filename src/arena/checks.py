"""Verifier primitives - the grader.

Checks read live cluster state and decide whether the task was actually done.
They never consult the agent's own account of what it did: a model that reports
success is making a claim, not providing evidence.

Two properties matter more than expressiveness here. A check must never raise on
unexpected cluster state - a grader that crashes scores nothing - and every
failure must name what it wanted and what it found, because that text is the
diagnosis in the results table.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_MISSING = object()
_INDEX = re.compile(r"^(.*?)\[(\d+)\]$")


@runtime_checkable
class ClusterView(Protocol):
    """The read surface a check needs. Satisfied by the live kubectl view."""

    def get(self, kind: str, namespace: str | None, name: str) -> dict[str, Any] | None: ...

    def list(
        self, kind: str, namespace: str | None, selector: dict[str, str] | None = None
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    description: str
    detail: str


class Check(Protocol):
    def evaluate(self, cluster: ClusterView) -> CheckResult: ...


def _traverse(root: Any, path: str) -> Any:
    """Walk a dotted path with optional ``[n]`` indices, or return a sentinel.

    Returns ``_MISSING`` rather than raising, so a check on state that never
    materialised reports a failure instead of aborting the run.
    """
    current = root
    for segment in path.split("."):
        index = None
        if match := _INDEX.match(segment):
            segment, index = match.group(1), int(match.group(2))

        if segment:
            if not isinstance(current, dict) or segment not in current:
                return _MISSING
            current = current[segment]

        if index is not None:
            if not isinstance(current, Sequence) or isinstance(current, str):
                return _MISSING
            if index >= len(current):
                return _MISSING
            current = current[index]

    return current


def _is_ready(pod: dict[str, Any]) -> bool:
    conditions = pod.get("status", {}).get("conditions", [])
    return any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)


@dataclass(frozen=True)
class PodsReady:
    """Exactly ``count`` pods matching ``selector`` are Ready.

    Exact rather than at-least: an agent that "fixes" a 2-replica deployment by
    scaling it to 10 has not fixed it.
    """

    namespace: str
    selector: dict[str, str]
    count: int

    def evaluate(self, cluster: ClusterView) -> CheckResult:
        selector = dict(self.selector)
        pods = cluster.list("Pod", self.namespace, selector)
        ready = [p for p in pods if _is_ready(p)]
        labels = ",".join(f"{k}={v}" for k, v in selector.items())
        description = f"{self.count} pod(s) matching {labels} are Ready in {self.namespace}"

        if len(ready) == self.count:
            return CheckResult(True, description, f"{len(ready)} ready, as expected")
        return CheckResult(
            False,
            description,
            f"expected {self.count} Ready pod(s), found {len(ready)} "
            f"Ready out of {len(pods)} matching pod(s)",
        )


@dataclass(frozen=True)
class FieldEquals:
    """A field at ``path`` on a named resource equals ``expected``."""

    kind: str
    namespace: str | None
    name: str
    path: str
    expected: Any

    def evaluate(self, cluster: ClusterView) -> CheckResult:
        target = f"{self.kind.lower()}/{self.name}"
        description = f"{target} {self.path} == {self.expected!r}"

        resource = cluster.get(self.kind, self.namespace, self.name)
        if resource is None:
            return CheckResult(
                False, description, f"resource {target} not found in namespace {self.namespace}"
            )

        actual = _traverse(resource, self.path)
        if actual is _MISSING:
            return CheckResult(
                False, description, f"{self.path} is not set on {target}"
            )
        if actual != self.expected:
            return CheckResult(
                False, description, f"expected {self.expected!r}, found {actual!r}"
            )
        return CheckResult(True, description, f"{self.path} == {actual!r}")


@dataclass(frozen=True)
class ResourceAbsent:
    """A resource no longer exists - for scenarios whose fix is a removal."""

    kind: str
    namespace: str | None
    name: str

    def evaluate(self, cluster: ClusterView) -> CheckResult:
        target = f"{self.kind.lower()}/{self.name}"
        description = f"{target} is absent from {self.namespace}"

        if cluster.get(self.kind, self.namespace, self.name) is None:
            return CheckResult(True, description, f"{target} is gone")
        return CheckResult(False, description, f"{target} still exists")


@dataclass(frozen=True)
class VerificationOutcome:
    results: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    def summary(self) -> str:
        if self.passed:
            return f"all {len(self.results)} check(s) passed"
        failures = [r for r in self.results if not r.passed]
        lines = [f"{len(failures)} of {len(self.results)} check(s) failed:"]
        lines += [f"  - {r.description}: {r.detail}" for r in failures]
        return "\n".join(lines)


class Verification:
    """All checks for one scenario. Passes only if every check passes."""

    def __init__(self, checks: Sequence[Check]):
        if not checks:
            raise ValueError(
                "a scenario must declare at least one check; an empty verification "
                "would pass every agent, including one that did nothing"
            )
        self.checks = list(checks)

    def evaluate(self, cluster: ClusterView) -> VerificationOutcome:
        # Every check runs even after one fails: diagnosis needs the full picture.
        return VerificationOutcome([check.evaluate(cluster) for check in self.checks])
