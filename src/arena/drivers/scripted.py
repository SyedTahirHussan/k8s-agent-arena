"""A driver that replays a fixed list of kubectl commands.

Two uses. It gives the harness deterministic end-to-end tests that cost nothing
and never flake on model nondeterminism. And it provides a second baseline: the
known-correct fix, replayed, which shows what a scenario's ceiling looks like -
useful for confirming a scenario is solvable at all before blaming an agent.
"""

from __future__ import annotations

from collections.abc import Sequence

from arena.drivers.base import Budget, ToolCall, ToolSurface, Transcript


class ScriptedDriver:
    """Issues its commands in order until they run out or the budget does."""

    def __init__(self, commands: Sequence, name: str = "scripted"):
        # A step is either a bare argument list or a mapping carrying a manifest
        # to pipe to stdin, which is how a fix that must *create* a resource is
        # expressed.
        self.commands = [
            (list(step["args"]), step.get("manifest"))
            if isinstance(step, dict)
            else (list(step), None)
            for step in commands
        ]
        self.name = name

    def run(self, task: str, tools: ToolSurface, budget: Budget) -> Transcript:
        calls: list[ToolCall] = []
        stop_reason = "finished"

        for command, manifest in self.commands:
            if len(calls) >= budget.max_turns:
                stop_reason = "budget_exhausted"
                break
            output, failed = tools.invoke(command, stdin=manifest)
            calls.append(
                ToolCall(name="kubectl", arguments=command, result=output, failed=failed)
            )

        return Transcript(
            driver=self.name,
            calls=calls,
            stop_reason=stop_reason,
            summary=f"replayed {len(calls)} scripted command(s)",
        )
