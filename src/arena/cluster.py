"""Throwaway kind clusters: provision, snapshot, destroy.

Every run gets its own cluster with a randomly suffixed, arena-owned name. That
is not tidiness - it is the safety model. A fresh cluster per run means the agent
cannot inherit state from a previous attempt, and the guard can assert that the
only thing reachable is something the arena created moments ago.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from arena.guard import ARENA_PREFIX, arena_cluster_name, ensure_safe_context
from arena.kubectl import build_argv
from arena.state import Snapshot

#: Kinds captured for blast-radius accounting. Broad enough to catch collateral
#: damage, narrow enough that a snapshot is one API round trip.
SNAPSHOT_KINDS = (
    "namespaces,nodes,deployments,statefulsets,daemonsets,replicasets,pods,"
    "services,configmaps,secrets,persistentvolumeclaims,persistentvolumes,"
    "ingresses,horizontalpodautoscalers,jobs,cronjobs,serviceaccounts,"
    "resourcequotas,limitranges,networkpolicies,poddisruptionbudgets"
)

CREATE_TIMEOUT_SECONDS = 300


class ClusterError(RuntimeError):
    """Provisioning or talking to the throwaway cluster failed."""


@dataclass(frozen=True)
class ClusterHandle:
    """A live throwaway cluster."""

    name: str
    context: str


def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise ClusterError(f"{argv[0]} is not installed: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ClusterError(f"{' '.join(argv[:2])} timed out after {timeout}s") from exc


def list_clusters() -> list[str]:
    """Every kind cluster on this machine, arena-owned or not."""
    result = _run(["kind", "get", "clusters"], timeout=60)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def orphaned_clusters(clusters: list[str]) -> list[str]:
    """Filter to clusters the arena created and is therefore allowed to delete.

    Deliberately strict. A developer's own kind clusters must be untouchable, so
    only the `arena-<random>` names the arena mints are ever candidates.
    """
    return [
        name
        for name in clusters
        if name.startswith(ARENA_PREFIX) and len(name) > len(ARENA_PREFIX)
    ]


def create_cluster(name: str | None = None) -> ClusterHandle:
    """Create a kind cluster the arena owns."""
    name = name or arena_cluster_name()
    context = f"kind-{name}"

    # Refuse before spending five minutes on a cluster we would not be allowed
    # to touch afterwards.
    ensure_safe_context(context, name)

    try:
        result = _run(
            ["kind", "create", "cluster", "--name", name, "--wait", "90s"],
            timeout=CREATE_TIMEOUT_SECONDS,
        )
    except KeyboardInterrupt:
        # Ctrl-C here lands before any context manager is armed, so without this
        # the half-built cluster outlives the process that made it.
        delete_cluster(name)
        raise

    if result.returncode != 0:
        raise ClusterError(f"kind create cluster failed:\n{result.stderr.strip()}")

    return ClusterHandle(name=name, context=context)


def delete_cluster(name: str) -> None:
    """Destroy a cluster. Never raises - teardown must not mask a run's result."""
    ensure_safe_context(f"kind-{name}", name)
    _run(["kind", "delete", "cluster", "--name", name], timeout=120)


@contextmanager
def arena_cluster(keep: bool = False) -> Iterator[ClusterHandle]:
    """Provision a cluster for the duration of a run, then destroy it.

    Args:
        keep: Leave the cluster running afterwards for post-mortem inspection.
            The caller is then responsible for `kind delete cluster`.
    """
    handle = create_cluster()
    try:
        yield handle
    finally:
        if not keep:
            # `finally` rather than `except Exception`, so a Ctrl-C mid-run still
            # tears the cluster down instead of leaving it holding a container.
            delete_cluster(handle.name)


def apply_manifests(path: Path, context: str, namespace: str | None = None) -> None:
    """Apply a manifest file, creating the namespace first if one is named."""
    if namespace:
        create = _run(
            build_argv(["create", "namespace", namespace], context), timeout=60
        )
        if create.returncode != 0 and "already exists" not in create.stderr:
            raise ClusterError(f"could not create namespace {namespace}:\n{create.stderr.strip()}")

    args = ["apply", "-f", str(path)]
    if namespace:
        args += ["-n", namespace]

    result = _run(build_argv(args, context), timeout=120)
    if result.returncode != 0:
        raise ClusterError(f"kubectl apply failed:\n{result.stderr.strip()}")


def snapshot(context: str) -> Snapshot:
    """Fingerprint the cluster's current state for blast-radius accounting."""
    import json

    argv = build_argv(
        ["get", SNAPSHOT_KINDS, "-A", "-o", "json", "--ignore-not-found"], context
    )
    result = _run(argv, timeout=120)
    if result.returncode != 0:
        raise ClusterError(f"could not snapshot cluster:\n{result.stderr.strip()}")

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ClusterError(f"unreadable snapshot from kubectl: {exc}") from exc

    return Snapshot.from_items(payload.get("items", []))
