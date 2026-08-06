"""The safety rail.

The arena hands a language model a kubectl surface and scores how much damage it
does. That is only an acceptable experiment against a throwaway cluster the arena
itself created. Every test here describes a way that assumption could break.
"""

import pytest

from arena.guard import UnsafeContextError, arena_cluster_name, ensure_safe_context


def test_accepts_the_kind_context_for_the_cluster_we_provisioned():
    ensure_safe_context(active_context="kind-arena-a1b2c3", cluster="arena-a1b2c3")


def test_rejects_a_context_that_is_not_a_kind_cluster():
    with pytest.raises(UnsafeContextError):
        ensure_safe_context(active_context="prod-eks-eu-central-1", cluster="arena-a1b2c3")


def test_rejects_a_different_kind_cluster_on_the_same_machine():
    """Developers keep other kind clusters around. Ours is not the only one."""
    with pytest.raises(UnsafeContextError):
        ensure_safe_context(active_context="kind-my-other-project", cluster="arena-a1b2c3")


def test_rejects_a_context_that_merely_starts_with_our_cluster_name():
    """Substring matching would accept 'kind-arena-a1b2c3-prod'. Match exactly."""
    with pytest.raises(UnsafeContextError):
        ensure_safe_context(active_context="kind-arena-a1b2c3-prod", cluster="arena-a1b2c3")


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_rejects_a_missing_active_context(empty):
    """No current-context means kubectl falls back to whatever is default. Refuse."""
    with pytest.raises(UnsafeContextError):
        ensure_safe_context(active_context=empty, cluster="arena-a1b2c3")


def test_rejects_a_cluster_the_arena_does_not_own():
    """Defence in depth: even a matching context is refused if the cluster name is
    not one the arena could have minted. A caller cannot talk us into 'production'."""
    with pytest.raises(UnsafeContextError):
        ensure_safe_context(active_context="kind-production", cluster="production")


def test_error_names_what_it_saw_and_what_it_wanted():
    """A safety abort has to be diagnosable at a glance."""
    with pytest.raises(UnsafeContextError) as excinfo:
        ensure_safe_context(active_context="prod-eks-eu-central-1", cluster="arena-a1b2c3")

    message = str(excinfo.value)
    assert "prod-eks-eu-central-1" in message
    assert "kind-arena-a1b2c3" in message


def test_generated_cluster_names_are_arena_owned_and_unique():
    first, second = arena_cluster_name(), arena_cluster_name()

    assert first != second
    assert first.startswith("arena-")
    # A generated name must survive its own guard.
    ensure_safe_context(active_context=f"kind-{first}", cluster=first)
