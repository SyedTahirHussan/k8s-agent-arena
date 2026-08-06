"""The driver seam.

A driver is anything that can be handed a task and a kubectl surface and left to
work: Gemini, K8sGPT, a scripted replay, or a control that does nothing. Keeping
this interface thin is what makes the benchmark a comparison rather than a demo -
every agent is scored by identical machinery on identical scenarios.

Drivers report what they did. They do not report whether it worked; that is the
grader's job, and asking an agent to assess itself would measure its confidence
rather than its competence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

DEFAULT_MAX_TURNS = 25
DEFAULT_MAX_SECONDS = 600


@runtime_checkable
class ToolSurface(Protocol):
    """The kubectl surface exposed to an agent."""

    def invoke(self, args: list[str], stdin: str | None = None) -> tuple[str, bool]:
        """Run a kubectl command, optionally piping ``stdin`` to it."""
        ...


@dataclass(frozen=True)
class Budget:
    """Hard limits enforced by the harness, never delegated to the agent."""

    max_turns: int = DEFAULT_MAX_TURNS
    max_seconds: int = DEFAULT_MAX_SECONDS

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {self.max_turns}")
        if self.max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {self.max_seconds}")


@dataclass(frozen=True)
class ToolCall:
    """One kubectl command an agent issued, and what came back."""

    name: str
    arguments: list[str]
    result: str
    failed: bool = False


@dataclass(frozen=True)
class Transcript:
    """What an agent did, for scoring and for the audit trail.

    Everything here is observed by the harness rather than self-reported, apart
    from ``summary``, which is the agent's own closing account and is recorded
    for diagnosis only - never for scoring.
    """

    driver: str
    calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "finished"
    summary: str = ""
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def failed_calls(self) -> int:
        return sum(1 for call in self.calls if call.failed)


class AgentDriver(Protocol):
    """Anything that can attempt a scenario."""

    name: str

    def run(self, task: str, tools: ToolSurface, budget: Budget) -> Transcript: ...
