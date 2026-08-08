"""Pacing requests to stay inside a provider's window.

Both free tiers used here allow five requests a minute, and one agent run makes
six to eight. Firing as fast as possible therefore guarantees a 429 partway
through every run, and waiting for the refusal costs more than avoiding it: the
API states a wait measured from its own window rather than from the client's, so
each throttled request pays a penalty a client-side window would not have
incurred.

The window is a rolling one, not a fixed slot. Enforcing an even 12-second gap
would be simpler and would also throw away the burst the provider genuinely
allows, which matters when a run's first few calls are the cheap diagnostic ones.

Backoff on 429 stays where it is. This reduces how often that path is taken; it
cannot replace it, because the window is per key and nothing here can see another
process spending the same quota.
"""

from __future__ import annotations

import time
from collections import deque

#: What both providers' free tiers allow. Cerebras states it in
#: `x-ratelimit-limit-requests-minute`; Gemini documents the same figure.
DEFAULT_MAX_REQUESTS = 5
DEFAULT_WINDOW_SECONDS = 60.0
#: Waking at the exact instant the window opens tends to land just inside it and
#: be refused, which is the outcome this exists to avoid.
MARGIN_SECONDS = 1.0


class RateLimiter:
    """Delays requests so no more than ``max_requests`` start per window.

    One limiter is meant to be shared by every run in a sweep: the allowance
    belongs to the API key, so a limiter per run would let each scenario open
    with a fresh burst and be throttled for it.
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        per_seconds: float = DEFAULT_WINDOW_SECONDS,
        now=time.monotonic,
        sleep=time.sleep,
    ):
        if max_requests <= 0:
            raise ValueError(f"max_requests must be positive, got {max_requests}")
        if per_seconds <= 0:
            raise ValueError(f"per_seconds must be positive, got {per_seconds}")

        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.total_waited = 0.0
        self._now = now
        self._sleep = sleep
        self._started: deque[float] = deque()

    def acquire(self) -> float:
        """Block until another request may start. Returns the seconds waited.

        The wait is reported rather than swallowed because a sweep attributes
        wall clock to the models under test, and time spent pacing is the
        harness's, not theirs.
        """
        self._forget_old()

        waited = 0.0
        if len(self._started) >= self.max_requests:
            oldest = self._started[0]
            waited = max(0.0, oldest + self.per_seconds - self._now()) + MARGIN_SECONDS
            self._sleep(waited)
            self.total_waited += waited
            # The sleep moved the clock, so the window has to be re-pruned - one
            # that is not stalls on every subsequent call.
            self._forget_old()

        self._started.append(self._now())
        return waited

    def _forget_old(self) -> None:
        cutoff = self._now() - self.per_seconds
        while self._started and self._started[0] <= cutoff:
            self._started.popleft()
