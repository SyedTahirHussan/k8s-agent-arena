"""Pacing requests to stay inside a provider's window.

Both free tiers allow five requests a minute and one agent run makes six to
eight, so a run that fires as fast as it can is throttled by design. Waiting for
the 429 and then honouring `retry-after` works, but it pays for the refusal every
time: the API states a wait measured from its own window, which is longer than
the wait the client could have computed, and a sweep spends that difference
fifteen times over.

Pacing up front removes the refusal. What it must not do is pace more than
necessary - a fixed sleep between requests would throw away the burst the window
genuinely allows, and a sweep is slow enough already.
"""

import pytest

from arena.ratelimit import RateLimiter


class Clock:
    """A clock that only moves when something sleeps, or when a test says so."""

    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def a_limiter(clock, max_requests=5, per_seconds=60.0):
    return RateLimiter(
        max_requests=max_requests,
        per_seconds=per_seconds,
        now=clock,
        sleep=clock.sleep,
    )


def test_a_burst_up_to_the_limit_is_not_delayed():
    """The window allows five at once, and pretending otherwise wastes a minute."""
    clock = Clock()
    limiter = a_limiter(clock)

    for _ in range(5):
        limiter.acquire()

    assert clock.slept == []
    assert clock.now == 0.0


def test_one_request_past_the_limit_waits_for_the_window():
    clock = Clock()
    limiter = a_limiter(clock)
    for _ in range(5):
        limiter.acquire()

    limiter.acquire()

    assert clock.slept, "the sixth request has to wait"
    assert clock.now >= 60.0


def test_it_waits_only_until_the_oldest_request_leaves_the_window():
    """Sleeping a whole window when nine seconds would do is a minute per run."""
    clock = Clock()
    limiter = a_limiter(clock)
    for _ in range(5):
        limiter.acquire()
    clock.advance(50.0)

    limiter.acquire()

    # 10s remained on the oldest request; a small margin on top is fine.
    assert 10.0 <= clock.slept[0] <= 12.0


def test_requests_spread_out_are_never_delayed():
    clock = Clock()
    limiter = a_limiter(clock)

    for _ in range(20):
        limiter.acquire()
        clock.advance(13.0)

    assert clock.slept == []


def test_the_window_keeps_sliding_after_a_wait():
    """A limiter that forgets to prune after sleeping stalls on every later call."""
    clock = Clock()
    limiter = a_limiter(clock)
    for _ in range(6):
        limiter.acquire()
    waits_so_far = len(clock.slept)

    clock.advance(60.0)
    limiter.acquire()

    assert len(clock.slept) == waits_so_far


def test_it_reports_how_long_it_waited():
    """The sweep attributes wall clock to models, so pacing has to be visible."""
    clock = Clock()
    limiter = a_limiter(clock)
    for _ in range(5):
        limiter.acquire()

    waited = limiter.acquire()

    assert waited > 0
    assert limiter.total_waited == pytest.approx(waited)


def test_a_request_that_did_not_wait_reports_zero():
    clock = Clock()
    limiter = a_limiter(clock)

    assert limiter.acquire() == 0.0
    assert limiter.total_waited == 0.0


def test_a_higher_allowance_paces_less():
    """A paid tier must not be throttled by the harness's own free-tier defaults."""
    clock = Clock()
    limiter = a_limiter(clock, max_requests=30)

    for _ in range(30):
        limiter.acquire()

    assert clock.slept == []


@pytest.mark.parametrize("bad", [0, -1])
def test_a_nonsensical_allowance_is_rejected(bad):
    with pytest.raises(ValueError, match="max_requests"):
        RateLimiter(max_requests=bad)


@pytest.mark.parametrize("bad", [0, -60.0])
def test_a_nonsensical_window_is_rejected(bad):
    with pytest.raises(ValueError, match="per_seconds"):
        RateLimiter(per_seconds=bad)


def test_one_limiter_is_shared_across_the_runs_of_a_sweep():
    """The quota belongs to the key, not the run. A per-run limiter would burst
    five requests at the start of every scenario and be refused every time."""
    clock = Clock()
    limiter = a_limiter(clock)

    for _ in range(3):  # three "runs", each making two requests
        for _ in range(2):
            limiter.acquire()

    assert clock.slept, "the sixth request across the sweep still has to wait"
