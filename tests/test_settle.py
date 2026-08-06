"""Grading has to wait for the cluster to catch up.

An agent that issues a correct `kubectl set image` has fixed the cluster, but the
rollout takes seconds and the old pods are still terminating. Grading the instant
the agent stops would fail correct answers and make the benchmark measure
reaction time instead of competence.

Equally, a scenario the agent never fixed must not hold the run open for the full
timeout on every repeat, so the wait ends the moment the checks pass.
"""

from arena.checks import CheckResult, VerificationOutcome
from arena.settle import evaluate_until_passing


class FakeClock:
    """A clock the test drives, so no test ever sleeps for real."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Eventually:
    """A verification that starts failing and passes after N attempts."""

    def __init__(self, passes_on_attempt: int):
        self.passes_on_attempt = passes_on_attempt
        self.attempts = 0

    def evaluate(self, cluster):
        self.attempts += 1
        passed = self.attempts >= self.passes_on_attempt
        return VerificationOutcome([CheckResult(passed, "rollout complete", "")])


def test_a_cluster_that_is_already_correct_is_not_waited_on():
    clock = FakeClock()
    verification = Eventually(passes_on_attempt=1)

    outcome = evaluate_until_passing(
        verification, cluster=None, timeout_seconds=60, clock=clock
    )

    assert outcome.passed
    assert clock.sleeps == []


def test_it_keeps_checking_while_the_rollout_finishes():
    clock = FakeClock()
    verification = Eventually(passes_on_attempt=4)

    outcome = evaluate_until_passing(
        verification, cluster=None, timeout_seconds=60, clock=clock
    )

    assert outcome.passed
    assert verification.attempts == 4


def test_it_gives_up_at_the_timeout():
    clock = FakeClock()
    verification = Eventually(passes_on_attempt=10_000)

    outcome = evaluate_until_passing(
        verification, cluster=None, timeout_seconds=10, clock=clock
    )

    assert not outcome.passed
    assert clock.time() <= 10 + 5  # bounded by the timeout, give or take one poll


def test_the_failure_it_reports_is_the_most_recent_one():
    """Stale diagnosis from the first poll would mislead whoever reads the table."""
    clock = FakeClock()

    class Degrading:
        def __init__(self):
            self.attempts = 0

        def evaluate(self, cluster):
            self.attempts += 1
            return VerificationOutcome(
                [CheckResult(False, "check", f"attempt {self.attempts}")]
            )

    outcome = evaluate_until_passing(
        Degrading(), cluster=None, timeout_seconds=10, clock=clock
    )

    assert "attempt" in outcome.results[0].detail
    assert outcome.results[0].detail != "attempt 1"


def test_a_zero_timeout_still_checks_once():
    """Otherwise a misconfigured timeout silently reports every agent as failing."""
    verification = Eventually(passes_on_attempt=1)

    outcome = evaluate_until_passing(
        verification, cluster=None, timeout_seconds=0, clock=FakeClock()
    )

    assert outcome.passed
    assert verification.attempts == 1
