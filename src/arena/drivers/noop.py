"""The control condition: an agent that does nothing.

Every benchmark needs a floor. Whatever score this driver earns on a scenario is
what that scenario awards for inaction - so a scenario the noop driver passes is
a broken scenario, not an easy one. Run it first when adding scenarios.
"""

from __future__ import annotations

from arena.drivers.base import Budget, ToolSurface, Transcript


class NoopDriver:
    """Touches nothing and reports honestly."""

    name = "noop"

    def run(self, task: str, tools: ToolSurface, budget: Budget) -> Transcript:
        return Transcript(
            driver=self.name,
            stop_reason="finished",
            summary="control condition: no action taken",
        )
