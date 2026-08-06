"""The kubectl surface handed to the agent.

The kubeconfig guard pins the *ambient* context, but an agent writes its own
argv, and `kubectl --context prod get pods` never consults ambient context. Any
argument that can retarget the client therefore has to be refused before a
process is spawned - otherwise the sandbox is advisory.

What is deliberately *not* restricted is destruction. `delete`, `scale` and
`patch` all run: the benchmark measures how much an agent breaks, which requires
letting it break things. Blast radius is a measurement, not a guardrail.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

#: Flags that point the client at a different cluster or identity. Each defeats
#: the ambient-context guard on its own.
BANNED_FLAGS = frozenset({
    "--context",
    "--kubeconfig",
    "--server",
    "--token",
    "--as",
    "--as-group",
    "--user",
    "--cluster",
    "--client-certificate",
    "--client-key",
    "--certificate-authority",
    "--insecure-skip-tls-verify",
})

#: Subcommands that move the target or open a path off the cluster.
BANNED_SUBCOMMANDS = frozenset({"config", "proxy", "port-forward"})

#: Tool output cap. `kubectl get -A -o yaml` can return megabytes, which blows
#: up both the context window and the bill.
DEFAULT_OUTPUT_LIMIT = 12_000


class UnsafeCommandError(ValueError):
    """The agent proposed a command that could escape the throwaway cluster."""


def build_argv(command: list[str], context: str) -> list[str]:
    """Turn an agent's kubectl arguments into a safe argv pinned to ``context``.

    Raises:
        UnsafeCommandError: If the command could address another cluster.
    """
    if not command:
        raise UnsafeCommandError("empty kubectl command")

    for argument in command:
        # Compare the flag name exactly, so `--show-labels` is not caught by the
        # ban on `--server` and friends.
        flag = argument.split("=", 1)[0]
        if flag in BANNED_FLAGS:
            raise UnsafeCommandError(
                f"refusing kubectl argument {argument!r}: {flag} can retarget the "
                "client at another cluster, which would bypass the context guard"
            )

    subcommand = next((arg for arg in command if not arg.startswith("-")), None)
    if subcommand in BANNED_SUBCOMMANDS:
        raise UnsafeCommandError(
            f"refusing kubectl subcommand {subcommand!r}: it can move the target "
            "cluster or open a network path off it"
        )

    return ["kubectl", "--context", context, *command]


def truncate_output(text: str, limit: int = DEFAULT_OUTPUT_LIMIT) -> str:
    """Bound tool output, keeping the head where the identifying detail lives."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... [truncated, {omitted} more characters]"


@dataclass
class KubectlSurface:
    """Runs agent-issued kubectl commands against the arena's cluster."""

    context: str
    timeout_seconds: int = 60
    output_limit: int = DEFAULT_OUTPUT_LIMIT

    def invoke(self, args: list[str], stdin: str | None = None) -> tuple[str, bool]:
        """Run one command. Returns ``(output, failed)``; never raises on failure.

        An agent issuing invalid kubectl is data about the agent, not an error
        in the harness, so non-zero exits come back as ordinary output.
        """
        try:
            argv = build_argv(args, self.context)
        except UnsafeCommandError as exc:
            return (f"refused: {exc}", True)

        try:
            completed = subprocess.run(
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return (f"command timed out after {self.timeout_seconds}s", True)
        except OSError as exc:
            return (f"could not run kubectl: {exc}", True)

        output = completed.stdout + completed.stderr
        return (truncate_output(output, self.output_limit), completed.returncode != 0)


@dataclass
class LiveClusterView:
    """Read-only cluster access for the grader, independent of the agent's surface."""

    context: str
    timeout_seconds: int = 60

    def _fetch(self, args: list[str]) -> list[dict[str, Any]]:
        argv = build_argv([*args, "-o", "json"], self.context)
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=self.timeout_seconds
        )
        if completed.returncode != 0:
            return []
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return []
        if payload.get("kind", "").endswith("List"):
            return payload.get("items", [])
        return [payload]

    def get(self, kind: str, namespace: str | None, name: str) -> dict[str, Any] | None:
        scope = ["-n", namespace] if namespace else ["--all-namespaces=false"]
        items = self._fetch([*["get", kind, name], *scope])
        return items[0] if items else None

    def list(
        self, kind: str, namespace: str | None, selector: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        args = ["get", kind]
        args += ["-n", namespace] if namespace else ["-A"]
        if selector:
            args += ["-l", ",".join(f"{k}={v}" for k, v in selector.items())]
        return self._fetch(args)
