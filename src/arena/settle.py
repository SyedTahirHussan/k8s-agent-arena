"""Wait for the cluster to catch up before grading.

A correct `kubectl set image` is correct the moment it is issued, but the rollout
takes seconds and the old pods are still terminating. Grading immediately would
fail correct answers and turn the benchmark into a measure of reaction time.

So checks are polled rather than sampled: the wait ends as soon as they pass, and
a run that was never fixed still costs only the timeout.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

DEFAULT_POLL_SECONDS = 5.0


class Clock(Protocol):
    def time(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class _RealClock:
    def time(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def evaluate_until_passing(
    verification: Any,
    cluster: Any,
    timeout_seconds: float,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    clock: Clock | None = None,
) -> Any:
    """Poll ``verification`` until it passes or ``timeout_seconds`` elapses.

    Returns the most recent outcome, so a reported failure reflects the cluster's
    final state rather than a stale first look.
    """
    clock = clock or _RealClock()
    deadline = clock.time() + timeout_seconds

    while True:
        outcome = verification.evaluate(cluster)
        if outcome.passed:
            return outcome
        if clock.time() >= deadline:
            return outcome
        clock.sleep(poll_seconds)
