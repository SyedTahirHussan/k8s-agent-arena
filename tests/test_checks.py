"""Verifier primitives.

A benchmark is only as trustworthy as its grader. These checks decide whether an
agent actually fixed the cluster, so they read live cluster state and never take
the agent's word for anything. Every failure has to say what it wanted and what
it found, because that text ends up in the results table.
"""

import pytest

from arena.checks import FieldEquals, PodsReady, ResourceAbsent, Verification
from tests.support import InMemoryCluster, deployment, pod


# --- PodsReady ---------------------------------------------------------------

def test_passes_when_the_expected_number_of_pods_are_ready():
    cluster = InMemoryCluster([
        pod("web-1", labels={"app": "web"}),
        pod("web-2", labels={"app": "web"}),
    ])

    result = PodsReady(namespace="default", selector={"app": "web"}, count=2).evaluate(cluster)

    assert result.passed


def test_fails_when_a_pod_exists_but_is_not_ready():
    """Scheduled is not running. ImagePullBackOff pods exist and are useless."""
    cluster = InMemoryCluster([
        pod("web-1", labels={"app": "web"}, ready=True),
        pod("web-2", labels={"app": "web"}, ready=False),
    ])

    result = PodsReady(namespace="default", selector={"app": "web"}, count=2).evaluate(cluster)

    assert not result.passed
    assert "1" in result.detail and "2" in result.detail


def test_ignores_pods_that_do_not_match_the_selector():
    cluster = InMemoryCluster([
        pod("web-1", labels={"app": "web"}),
        pod("billing-1", labels={"app": "billing"}),
    ])

    result = PodsReady(namespace="default", selector={"app": "web"}, count=1).evaluate(cluster)

    assert result.passed


def test_fails_when_no_pods_exist_at_all():
    result = PodsReady(namespace="default", selector={"app": "web"}, count=1).evaluate(
        InMemoryCluster([])
    )

    assert not result.passed


def test_extra_ready_pods_do_not_satisfy_an_exact_count():
    """An agent that scales to 10 has not fixed a 2-replica deployment."""
    cluster = InMemoryCluster([
        pod(f"web-{i}", labels={"app": "web"}) for i in range(5)
    ])

    result = PodsReady(namespace="default", selector={"app": "web"}, count=2).evaluate(cluster)

    assert not result.passed


# --- FieldEquals -------------------------------------------------------------

def test_reads_a_nested_field_through_a_dotted_path():
    cluster = InMemoryCluster([deployment("web", replicas=3)])

    result = FieldEquals(
        kind="Deployment", namespace="default", name="web",
        path="spec.replicas", expected=3,
    ).evaluate(cluster)

    assert result.passed


def test_indexes_into_lists_along_the_path():
    cluster = InMemoryCluster([deployment("web", image="nginx:1.29-alpine")])

    result = FieldEquals(
        kind="Deployment", namespace="default", name="web",
        path="spec.template.spec.containers[0].image", expected="nginx:1.29-alpine",
    ).evaluate(cluster)

    assert result.passed


def test_reports_both_expected_and_actual_when_a_field_differs():
    cluster = InMemoryCluster([deployment("web", image="nginx:doesnotexist")])

    result = FieldEquals(
        kind="Deployment", namespace="default", name="web",
        path="spec.template.spec.containers[0].image", expected="nginx:1.29-alpine",
    ).evaluate(cluster)

    assert not result.passed
    assert "nginx:doesnotexist" in result.detail
    assert "nginx:1.29-alpine" in result.detail


def test_a_missing_path_fails_rather_than_raising():
    """A grader that crashes on unexpected state scores nothing at all."""
    cluster = InMemoryCluster([deployment("web")])

    result = FieldEquals(
        kind="Deployment", namespace="default", name="web",
        path="spec.template.spec.containers[0].resources.limits.memory", expected="256Mi",
    ).evaluate(cluster)

    assert not result.passed
    assert "not set" in result.detail.lower() or "missing" in result.detail.lower()


def test_an_out_of_range_list_index_fails_rather_than_raising():
    cluster = InMemoryCluster([deployment("web")])

    result = FieldEquals(
        kind="Deployment", namespace="default", name="web",
        path="spec.template.spec.containers[3].image", expected="nginx",
    ).evaluate(cluster)

    assert not result.passed


def test_a_missing_resource_fails_rather_than_raising():
    result = FieldEquals(
        kind="Deployment", namespace="default", name="ghost",
        path="spec.replicas", expected=1,
    ).evaluate(InMemoryCluster([]))

    assert not result.passed
    assert "ghost" in result.detail


# --- ResourceAbsent ----------------------------------------------------------

def test_absent_passes_when_the_resource_is_gone():
    result = ResourceAbsent(kind="Pod", namespace="default", name="stuck").evaluate(
        InMemoryCluster([])
    )

    assert result.passed


def test_absent_fails_when_the_resource_is_still_there():
    result = ResourceAbsent(kind="Pod", namespace="default", name="stuck").evaluate(
        InMemoryCluster([pod("stuck")])
    )

    assert not result.passed


# --- Verification ------------------------------------------------------------

def test_verification_passes_only_when_every_check_passes():
    cluster = InMemoryCluster([deployment("web", replicas=2), pod("web-1", labels={"app": "web"})])

    verification = Verification([
        PodsReady(namespace="default", selector={"app": "web"}, count=1),
        FieldEquals(kind="Deployment", namespace="default", name="web",
                    path="spec.replicas", expected=2),
    ])

    assert verification.evaluate(cluster).passed


def test_verification_fails_and_names_the_check_that_broke():
    cluster = InMemoryCluster([deployment("web", replicas=99)])

    verification = Verification([
        FieldEquals(kind="Deployment", namespace="default", name="web",
                    path="spec.replicas", expected=2),
    ])
    outcome = verification.evaluate(cluster)

    assert not outcome.passed
    assert "spec.replicas" in outcome.summary()


def test_verification_runs_every_check_even_after_one_fails():
    """Partial credit and diagnosis both need the full picture, not the first error."""
    cluster = InMemoryCluster([])

    verification = Verification([
        FieldEquals(kind="Deployment", namespace="default", name="a", path="spec.replicas", expected=1),
        FieldEquals(kind="Deployment", namespace="default", name="b", path="spec.replicas", expected=1),
    ])
    outcome = verification.evaluate(cluster)

    assert len(outcome.results) == 2
    assert outcome.passed_count == 0


def test_an_empty_verification_is_a_configuration_error():
    """A scenario that checks nothing would silently pass every agent."""
    with pytest.raises(ValueError):
        Verification([])


# --- FieldPresent ------------------------------------------------------------
# Some fixes are outcome-based rather than prescriptive: "give it enough memory"
# has many right answers. FieldPresent lets a scenario require that a setting
# still exists without dictating its value - so raising a limit passes and
# deleting the limit outright does not.

def test_present_passes_when_the_field_is_set():
    from arena.checks import FieldPresent

    cluster = InMemoryCluster([deployment("web", replicas=2)])

    result = FieldPresent(
        kind="Deployment", namespace="default", name="web", path="spec.replicas"
    ).evaluate(cluster)

    assert result.passed


def test_present_fails_when_the_field_was_removed():
    from arena.checks import FieldPresent

    cluster = InMemoryCluster([deployment("web")])

    result = FieldPresent(
        kind="Deployment", namespace="default", name="web",
        path="spec.template.spec.containers[0].resources.limits.memory",
    ).evaluate(cluster)

    assert not result.passed


def test_present_fails_when_the_resource_is_gone():
    from arena.checks import FieldPresent

    result = FieldPresent(
        kind="Deployment", namespace="default", name="ghost", path="spec.replicas"
    ).evaluate(InMemoryCluster([]))

    assert not result.passed


# --- PodRestartsBelow --------------------------------------------------------
# A crash-looping pod is Ready for a moment between restarts. Because grading
# polls until it passes, `pods_ready` alone will eventually catch that window and
# call a broken workload fixed - which is how the do-nothing control passed the
# OOM scenario. Stability has to be asserted separately from readiness.

def test_restarts_passes_when_the_pod_is_stable():
    from arena.checks import PodRestartsBelow

    cluster = InMemoryCluster([pod("cache-1", labels={"app": "cache"}, restarts=0)])

    result = PodRestartsBelow(
        namespace="default", selector={"app": "cache"}, max_restarts=3
    ).evaluate(cluster)

    assert result.passed


def test_restarts_fails_a_crashlooping_pod():
    from arena.checks import PodRestartsBelow

    cluster = InMemoryCluster([pod("cache-1", labels={"app": "cache"}, restarts=7)])

    result = PodRestartsBelow(
        namespace="default", selector={"app": "cache"}, max_restarts=3
    ).evaluate(cluster)

    assert not result.passed
    assert "7" in result.detail


def test_restarts_judges_by_the_worst_pod_not_the_average():
    from arena.checks import PodRestartsBelow

    cluster = InMemoryCluster([
        pod("cache-1", labels={"app": "cache"}, restarts=0),
        pod("cache-2", labels={"app": "cache"}, restarts=9),
    ])

    result = PodRestartsBelow(
        namespace="default", selector={"app": "cache"}, max_restarts=3
    ).evaluate(cluster)

    assert not result.passed


def test_restarts_fails_when_no_pod_exists_at_all():
    """No pods is not stability; it is absence. Passing vacuously would hide it."""
    from arena.checks import PodRestartsBelow

    result = PodRestartsBelow(
        namespace="default", selector={"app": "cache"}, max_restarts=3
    ).evaluate(InMemoryCluster([]))

    assert not result.passed


def test_restarts_treats_a_pod_without_container_status_as_zero():
    """A pod that has not started yet has not restarted yet."""
    from arena.checks import PodRestartsBelow

    bare = pod("cache-1", labels={"app": "cache"})
    bare["status"].pop("containerStatuses", None)

    result = PodRestartsBelow(
        namespace="default", selector={"app": "cache"}, max_restarts=3
    ).evaluate(InMemoryCluster([bare]))

    assert result.passed
