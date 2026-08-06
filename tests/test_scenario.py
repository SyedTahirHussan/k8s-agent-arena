"""Scenario definitions loaded from YAML.

A scenario is the unit of the benchmark: a broken cluster, a task written the way
a human would hand it over, and a grader. Scenarios are data rather than code so
that adding one is a pull request anybody can review.

The subtle part is `in_scope`. Fixing a Deployment causes Kubernetes to roll its
ReplicaSets and Pods - churn the agent caused but did not choose. Counting that as
collateral damage would make every successful repair look reckless, so scope
matching has to cover derived resources.
"""

import pytest

from arena.scenario import ScenarioError, load_scenario_text
from arena.state import ResourceRef

FULL = """
id: imagepull-backoff
title: Deployment stuck in ImagePullBackOff
namespace: arena-imagepull
manifests: broken.yaml
task: |
  The deployment `web` will not come up. Diagnose and fix it.
in_scope:
  - kind: Deployment
    name: web
verify:
  - type: pods_ready
    selector: {app: web}
    count: 2
  - type: field_equals
    kind: Deployment
    name: web
    path: spec.template.spec.containers[0].image
    expected: nginx:1.29-alpine
timeout_seconds: 300
"""


def test_loads_every_declared_field():
    scenario = load_scenario_text(FULL)

    assert scenario.id == "imagepull-backoff"
    assert scenario.title == "Deployment stuck in ImagePullBackOff"
    assert scenario.namespace == "arena-imagepull"
    assert scenario.manifests == "broken.yaml"
    assert "will not come up" in scenario.task
    assert scenario.timeout_seconds == 300
    assert len(scenario.verification.checks) == 2


def test_checks_inherit_the_scenario_namespace():
    """Repeating the namespace on every check invites a typo that silently passes."""
    scenario = load_scenario_text(FULL)

    assert all(check.namespace == "arena-imagepull" for check in scenario.verification.checks)


def test_timeout_has_a_default():
    scenario = load_scenario_text(FULL.replace("timeout_seconds: 300", ""))

    assert scenario.timeout_seconds > 0


@pytest.mark.parametrize("missing", ["id", "task", "namespace"])
def test_a_missing_required_field_names_itself(missing):
    text = "\n".join(line for line in FULL.splitlines() if not line.startswith(f"{missing}:"))

    with pytest.raises(ScenarioError) as excinfo:
        load_scenario_text(text)

    assert missing in str(excinfo.value)


def test_an_unknown_check_type_lists_the_valid_ones():
    text = FULL.replace("type: pods_ready", "type: vibes_check")

    with pytest.raises(ScenarioError) as excinfo:
        load_scenario_text(text)

    message = str(excinfo.value)
    assert "vibes_check" in message
    assert "pods_ready" in message


def test_a_scenario_with_no_checks_is_rejected():
    text = FULL.split("verify:")[0] + "verify: []\n"

    with pytest.raises(ScenarioError):
        load_scenario_text(text)


def test_a_check_missing_a_required_argument_is_rejected():
    text = FULL.replace("    count: 2", "")

    with pytest.raises(ScenarioError):
        load_scenario_text(text)


# --- scope matching ----------------------------------------------------------

def test_the_named_resource_is_in_scope():
    scenario = load_scenario_text(FULL)

    assert scenario.in_scope(
        ResourceRef("apps/v1", "Deployment", "arena-imagepull", "web")
    )


def test_an_unrelated_resource_is_out_of_scope():
    scenario = load_scenario_text(FULL)

    assert not scenario.in_scope(
        ResourceRef("apps/v1", "Deployment", "arena-imagepull", "billing")
    )


def test_the_same_name_in_another_namespace_is_out_of_scope():
    scenario = load_scenario_text(FULL)

    assert not scenario.in_scope(ResourceRef("apps/v1", "Deployment", "kube-system", "web"))


def test_pods_and_replicasets_rolled_by_an_in_scope_workload_are_in_scope():
    """Repairing a Deployment necessarily churns what it owns. That is not damage."""
    text = FULL.replace("    name: web\n", '    name: "web*"\n')
    scenario = load_scenario_text(text)

    for kind, name in [
        ("ReplicaSet", "web-7d9f8b6c4"),
        ("Pod", "web-7d9f8b6c4-nk2xq"),
    ]:
        assert scenario.in_scope(ResourceRef("v1", kind, "arena-imagepull", name)), kind


def test_a_wildcard_does_not_leak_across_workloads():
    """`web*` must not quietly authorise touching `webhook-controller`."""
    text = FULL.replace("    name: web\n", '    name: "web-*"\n')
    scenario = load_scenario_text(text)

    assert not scenario.in_scope(
        ResourceRef("apps/v1", "Deployment", "arena-imagepull", "webhook-controller")
    )


def test_scope_can_be_restricted_by_kind():
    scenario = load_scenario_text(FULL)

    assert not scenario.in_scope(
        ResourceRef("v1", "Secret", "arena-imagepull", "web")
    )
