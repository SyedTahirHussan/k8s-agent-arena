"""The kubectl surface handed to the agent.

This is the sharp edge of the project. The guard pins the *ambient* context to a
throwaway cluster, but an agent composes its own argv - and `kubectl --context
prod get pods` ignores ambient context entirely. Anything that lets an argument
list address a different cluster defeats the guard, so those flags are rejected
before the process is ever spawned.

The second concern is duller but bites every run: `kubectl get -A -o yaml` can
return megabytes. Unbounded tool output blows up the context window and the bill.
"""

import pytest

from arena.kubectl import UnsafeCommandError, build_argv, truncate_output


CONTEXT = "kind-arena-a1b2c3"


# --- the command is pinned to our cluster ------------------------------------

def test_every_command_is_pinned_to_the_arena_context():
    argv = build_argv(["get", "pods"], context=CONTEXT)

    assert argv[0] == "kubectl"
    assert "--context" in argv
    assert argv[argv.index("--context") + 1] == CONTEXT
    assert argv[-2:] == ["get", "pods"]


@pytest.mark.parametrize(
    "escape",
    [
        ["--context", "prod-eks"],
        ["--context=prod-eks"],
        ["--kubeconfig", "/home/user/.kube/prod"],
        ["--kubeconfig=/home/user/.kube/prod"],
        ["--server", "https://10.0.0.1:6443"],
        ["--server=https://10.0.0.1:6443"],
        ["--token", "abc123"],
        ["--as", "system:admin"],
    ],
)
def test_flags_that_would_retarget_another_cluster_are_rejected(escape):
    """The kubeconfig guard is ambient; argv beats ambient. Close the hole here."""
    with pytest.raises(UnsafeCommandError):
        build_argv(["get", "pods", *escape], context=CONTEXT)


def test_the_config_subcommand_is_rejected_outright():
    """`kubectl config use-context prod` would move the target under our feet."""
    with pytest.raises(UnsafeCommandError):
        build_argv(["config", "use-context", "prod"], context=CONTEXT)


@pytest.mark.parametrize("escape_hatch", ["proxy", "port-forward"])
def test_commands_that_open_a_network_path_out_are_rejected(escape_hatch):
    with pytest.raises(UnsafeCommandError):
        build_argv([escape_hatch, "svc/web", "8080:80"], context=CONTEXT)


def test_rejection_explains_which_argument_was_the_problem():
    with pytest.raises(UnsafeCommandError) as excinfo:
        build_argv(["get", "pods", "--context=prod-eks"], context=CONTEXT)

    assert "--context" in str(excinfo.value)


def test_an_empty_command_is_rejected():
    with pytest.raises(UnsafeCommandError):
        build_argv([], context=CONTEXT)


# --- ordinary operations still work ------------------------------------------

@pytest.mark.parametrize(
    "command",
    [
        ["get", "pods", "-A"],
        ["describe", "pod", "web-1"],
        ["logs", "web-1", "--previous"],
        ["apply", "-f", "-"],
        ["patch", "deployment/web", "--type=json", "-p", "[]"],
        ["set", "image", "deployment/web", "web=nginx:1.29"],
        ["scale", "deployment/web", "--replicas=3"],
        ["delete", "pod", "web-1"],
        ["rollout", "restart", "deployment/web"],
    ],
)
def test_normal_administration_is_permitted(command):
    """Blast radius is measured, not prevented. Destructive commands must run."""
    argv = build_argv(command, context=CONTEXT)

    assert argv[-len(command):] == command


def test_a_flag_that_merely_contains_a_banned_word_is_allowed():
    """`--context` is banned; `--show-labels` and friends must not be collateral."""
    argv = build_argv(["get", "pods", "--show-labels"], context=CONTEXT)

    assert "--show-labels" in argv


# --- output is bounded -------------------------------------------------------

def test_short_output_is_returned_unchanged():
    assert truncate_output("web-1  Running", limit=100) == "web-1  Running"


def test_long_output_is_truncated_and_says_so():
    text = "\n".join(f"line {i}" for i in range(10_000))

    truncated = truncate_output(text, limit=500)

    assert len(truncated) < len(text)
    assert "truncated" in truncated.lower()


def test_truncation_keeps_the_start_of_the_output():
    """The head of a kubectl dump carries the identifying detail."""
    text = "IMPORTANT FIRST LINE\n" + "\n".join(f"line {i}" for i in range(10_000))

    assert "IMPORTANT FIRST LINE" in truncate_output(text, limit=500)


# --- applying manifests from stdin -------------------------------------------

def test_a_manifest_can_be_piped_to_apply():
    """`kubectl apply -f -` is how an agent creates a resource it must recreate."""
    argv = build_argv(["apply", "-f", "-"], context=CONTEXT)

    assert argv[-3:] == ["apply", "-f", "-"]


def test_the_surface_accepts_stdin_without_requiring_it():
    from arena.kubectl import KubectlSurface
    import inspect

    signature = inspect.signature(KubectlSurface.invoke)
    assert "stdin" in signature.parameters
    assert signature.parameters["stdin"].default is None
