"""Scenario definitions: a broken cluster, a task, and a grader.

Scenarios are YAML rather than Python so that contributing one is a reviewable
data change. Each declares the manifests that create the fault, the task text
handed to the agent verbatim, the checks that decide success, and the resources
the agent is allowed to touch while doing it.

That last part - `in_scope` - carries the subtlety. Repairing a Deployment makes
Kubernetes roll its ReplicaSets and Pods: churn the agent caused but never chose.
Scope therefore follows ownership chains, so a legitimate repair is not scored as
collateral damage while an unrelated Secret with a similar name still is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from arena.checks import (
    Check,
    FieldEquals,
    FieldPresent,
    PodsReady,
    ResourceAbsent,
    Verification,
)
from arena.state import ResourceRef

DEFAULT_TIMEOUT_SECONDS = 300

#: Kinds Kubernetes creates on a controller's behalf. Naming the owner in
#: `in_scope` implicitly covers what the owner rolls.
_DERIVED_KINDS: Mapping[str, frozenset[str]] = {
    "Deployment": frozenset({"ReplicaSet", "Pod"}),
    "StatefulSet": frozenset({"Pod", "ControllerRevision", "PersistentVolumeClaim"}),
    "DaemonSet": frozenset({"Pod", "ControllerRevision"}),
    "ReplicaSet": frozenset({"Pod"}),
    "CronJob": frozenset({"Job", "Pod"}),
    "Job": frozenset({"Pod"}),
}


class ScenarioError(ValueError):
    """A scenario file is malformed. Raised with enough detail to fix it."""


@dataclass(frozen=True)
class ScopeEntry:
    """One `in_scope` rule. ``name`` may be a glob; ``kind`` may be omitted."""

    name: str
    namespace: str
    kind: str | None = None

    def matches(self, ref: ResourceRef) -> bool:
        if ref.namespace != self.namespace:
            return False
        if not fnmatchcase(ref.name, self.name):
            return False
        if self.kind is None:
            return True
        return ref.kind == self.kind or ref.kind in _DERIVED_KINDS.get(self.kind, frozenset())


def _require(data: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in data or data[key] in (None, ""):
        raise ScenarioError(f"{where} is missing required field {key!r}")
    return data[key]


def _build_pods_ready(spec: Mapping[str, Any], namespace: str) -> Check:
    return PodsReady(
        namespace=spec.get("namespace", namespace),
        selector=dict(_require(spec, "selector", "check 'pods_ready'")),
        count=int(_require(spec, "count", "check 'pods_ready'")),
    )


def _build_field_equals(spec: Mapping[str, Any], namespace: str) -> Check:
    return FieldEquals(
        kind=_require(spec, "kind", "check 'field_equals'"),
        namespace=spec.get("namespace", namespace),
        name=_require(spec, "name", "check 'field_equals'"),
        path=_require(spec, "path", "check 'field_equals'"),
        expected=_require(spec, "expected", "check 'field_equals'"),
    )


def _build_field_present(spec: Mapping[str, Any], namespace: str) -> Check:
    return FieldPresent(
        kind=_require(spec, "kind", "check 'field_present'"),
        namespace=spec.get("namespace", namespace),
        name=_require(spec, "name", "check 'field_present'"),
        path=_require(spec, "path", "check 'field_present'"),
    )


def _build_resource_absent(spec: Mapping[str, Any], namespace: str) -> Check:
    return ResourceAbsent(
        kind=_require(spec, "kind", "check 'resource_absent'"),
        namespace=spec.get("namespace", namespace),
        name=_require(spec, "name", "check 'resource_absent'"),
    )


_CHECK_BUILDERS = {
    "pods_ready": _build_pods_ready,
    "field_equals": _build_field_equals,
    "field_present": _build_field_present,
    "resource_absent": _build_resource_absent,
}


@dataclass(frozen=True)
class Scenario:
    """One benchmark case."""

    id: str
    title: str
    namespace: str
    task: str
    verification: Verification
    scope: tuple[ScopeEntry, ...] = ()
    #: The known-good fix, replayed by the `solution` driver. Its purpose is to
    #: prove the scenario is solvable, so a failing agent can be told apart from
    #: a broken scenario.
    solution: tuple[dict[str, Any], ...] = ()
    manifests: str | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    source_dir: Path | None = field(default=None, compare=False)

    def in_scope(self, ref: ResourceRef) -> bool:
        """Is changing ``ref`` a sanctioned part of doing this task?"""
        return any(entry.matches(ref) for entry in self.scope)

    def manifests_path(self) -> Path | None:
        """The manifest file, resolved relative to the scenario that declared it."""
        if self.manifests is None:
            return None
        base = self.source_dir or Path.cwd()
        return base / self.manifests


def _parse(data: Mapping[str, Any], source_dir: Path | None) -> Scenario:
    if not isinstance(data, Mapping):
        raise ScenarioError("scenario must be a YAML mapping")

    where = "scenario"
    scenario_id = _require(data, "id", where)
    namespace = _require(data, "namespace", where)
    task = _require(data, "task", where)

    raw_checks = data.get("verify") or []
    checks = []
    for index, spec in enumerate(raw_checks):
        check_type = spec.get("type")
        if check_type not in _CHECK_BUILDERS:
            valid = ", ".join(sorted(_CHECK_BUILDERS))
            raise ScenarioError(
                f"scenario {scenario_id!r} check #{index + 1} has unknown type "
                f"{check_type!r}; valid types are: {valid}"
            )
        checks.append(_CHECK_BUILDERS[check_type](spec, namespace))

    try:
        verification = Verification(checks)
    except ValueError as exc:
        raise ScenarioError(f"scenario {scenario_id!r}: {exc}") from exc

    scope = tuple(
        ScopeEntry(
            name=_require(entry, "name", f"scenario {scenario_id!r} in_scope entry"),
            namespace=entry.get("namespace", namespace),
            kind=entry.get("kind"),
        )
        for entry in (data.get("in_scope") or [])
    )

    solution = tuple(
        {
            "args": list(_require(step, "args", f"scenario {scenario_id!r} solution step")),
            "manifest": step.get("manifest"),
        }
        for step in (data.get("solution") or [])
    )

    return Scenario(
        id=scenario_id,
        title=data.get("title") or scenario_id,
        namespace=namespace,
        task=task,
        verification=verification,
        scope=scope,
        solution=solution,
        manifests=data.get("manifests"),
        timeout_seconds=int(data.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS),
        source_dir=source_dir,
    )


def load_scenario_text(text: str, source_dir: Path | None = None) -> Scenario:
    """Parse a scenario from YAML text."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioError(f"invalid YAML: {exc}") from exc
    return _parse(data or {}, source_dir)


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a YAML file."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ScenarioError(f"cannot read scenario {path}: {exc}") from exc
    return load_scenario_text(text, source_dir=path.parent)
