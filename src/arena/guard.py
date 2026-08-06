"""Refuse to run against anything but a throwaway cluster the arena created.

The arena deliberately gives a language model a kubectl surface and then measures
what it disturbed. That experiment is only safe against a cluster we minted and
intend to destroy. A stray ``current-context`` in ``~/.kube/config`` is the whole
threat model, so this check is a hard abort rather than a warning.
"""

from __future__ import annotations

import secrets

#: Cluster names the arena mints. The guard refuses to vouch for anything else,
#: so a caller cannot talk the runner into pointing at "production".
ARENA_PREFIX = "arena-"


class UnsafeContextError(RuntimeError):
    """Raised when the active kubectl context is not our throwaway cluster."""


def arena_cluster_name() -> str:
    """Mint a fresh, arena-owned cluster name."""
    return f"{ARENA_PREFIX}{secrets.token_hex(3)}"


def ensure_safe_context(active_context: str | None, cluster: str) -> None:
    """Abort unless ``active_context`` is exactly the kind context for ``cluster``.

    Args:
        active_context: What ``kubectl config current-context`` reports.
        cluster: The kind cluster the arena provisioned for this run.

    Raises:
        UnsafeContextError: On any mismatch, including a missing context, a
            foreign kind cluster, or a cluster name the arena does not own.
    """
    if not cluster.startswith(ARENA_PREFIX) or cluster == ARENA_PREFIX:
        raise UnsafeContextError(
            f"refusing to operate on cluster {cluster!r}: the arena only manages "
            f"clusters named {ARENA_PREFIX}*, and only ones it created itself"
        )

    expected = f"kind-{cluster}"

    if active_context is None or not active_context.strip():
        raise UnsafeContextError(
            f"no active kubectl context; expected {expected!r}. Refusing to run "
            "rather than fall back to whatever kubectl considers default."
        )

    if active_context != expected:
        raise UnsafeContextError(
            f"active kubectl context is {active_context!r}, expected {expected!r}. "
            "Refusing to run: the arena only acts on the throwaway cluster it created."
        )
