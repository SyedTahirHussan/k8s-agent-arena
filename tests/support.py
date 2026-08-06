"""Test doubles that are real implementations, not mocks.

`InMemoryCluster` satisfies the same read protocol as the live kubectl-backed
view, so checks under test exercise their actual traversal logic rather than
asserting against a recorded call.
"""

from __future__ import annotations

from typing import Any


def pod(name, namespace="default", labels=None, ready=True, phase="Running", **fields):
    conditions = [{"type": "Ready", "status": "True" if ready else "False"}]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace, "labels": labels or {}},
        "status": {"phase": phase, "conditions": conditions},
        **fields,
    }


def deployment(name, namespace="default", image="nginx:1.29", replicas=1, **fields):
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "replicas": replicas,
            "template": {
                "spec": {"containers": [{"name": "web", "image": image}]}
            },
        },
        **fields,
    }


class RecordingTools:
    """A kubectl surface that records invocations instead of running them.

    Real enough to drive a driver: it returns canned output, can be told which
    commands should fail, and keeps the call log the tests assert against.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        failures: set[str] | None = None,
    ):
        self.responses = responses or {}
        self.failures = failures or set()
        self.calls: list[list[str]] = []

    def invoke(self, args: list[str]) -> tuple[str, bool]:
        """Run a kubectl command. Returns (output, failed)."""
        self.calls.append(list(args))
        key = " ".join(args)
        if key in self.failures:
            return (f"error: unknown command {key!r}", True)
        return (self.responses.get(key, ""), False)


class InMemoryCluster:
    """A cluster view backed by a list of resource dicts."""

    def __init__(self, items: list[dict[str, Any]] | None = None):
        self.items = list(items or [])

    def get(self, kind: str, namespace: str | None, name: str) -> dict[str, Any] | None:
        for item in self.items:
            metadata = item.get("metadata", {})
            if (
                item.get("kind") == kind
                and metadata.get("namespace") == namespace
                and metadata.get("name") == name
            ):
                return item
        return None

    def list(
        self, kind: str, namespace: str | None, selector: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        matches = []
        for item in self.items:
            metadata = item.get("metadata", {})
            if item.get("kind") != kind or metadata.get("namespace") != namespace:
                continue
            labels = metadata.get("labels", {})
            if selector and any(labels.get(k) != v for k, v in selector.items()):
                continue
            matches.append(item)
        return matches
