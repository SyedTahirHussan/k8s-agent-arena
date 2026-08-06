"""Cleaning up clusters an interrupted run left behind.

Ctrl-C during `kind create cluster` happens before the context manager's
teardown is armed, so the cluster survives the process that made it. Left alone
they accumulate, each holding a container and a couple of gigabytes.

The whole risk of a cleanup command is that it deletes something it should not.
The arena names every cluster it creates `arena-<random>`, and these tests exist
to pin down that nothing else is ever a candidate.
"""

from arena.cluster import orphaned_clusters


def test_it_finds_clusters_the_arena_created():
    assert orphaned_clusters(["arena-6a1b81", "arena-519f5c"]) == [
        "arena-6a1b81",
        "arena-519f5c",
    ]


def test_it_never_offers_to_delete_someone_elses_cluster():
    """A developer's own kind clusters must be untouchable."""
    assert orphaned_clusters(["kind", "my-project", "staging-local"]) == []


def test_it_picks_arena_clusters_out_of_a_mixed_list():
    assert orphaned_clusters(["my-project", "arena-6a1b81", "kind"]) == ["arena-6a1b81"]


def test_a_cluster_merely_starting_with_arena_is_not_ours():
    """`arena-` is the prefix; `arenaball` and `arena` alone are not arena clusters."""
    assert orphaned_clusters(["arenaball", "arena", "arena-ok"]) == ["arena-ok"]


def test_an_empty_cluster_list_yields_nothing():
    assert orphaned_clusters([]) == []
