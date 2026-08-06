"""The Gemini driver.

An agent loop is mostly bookkeeping, and the bookkeeping is where benchmarks go
wrong. Three things matter here and each has a test.

Token accounting must include `thoughts_token_count`: Gemini 3 thinks before it
answers and those tokens are billed as output. Omitting them would understate the
cost of exactly the models that reason hardest, which is the comparison the
benchmark exists to make.

Model turns must be appended to history verbatim, because Gemini 3 carries
encrypted thought signatures on its own parts and rebuilding them loses reasoning
context mid-task.

And an API failure has to be recorded as a failed run rather than raised, or one
500 mid-sweep destroys every result collected so far.
"""

import pytest

from arena.drivers.base import Budget
from arena.drivers.gemini import GeminiDriver
from tests.support import FakeGemini, RecordingTools, model_calls, model_says


def test_it_runs_the_command_the_model_asks_for():
    tools = RecordingTools()
    client = FakeGemini([model_calls(["get", "pods", "-n", "arena-x"]), model_says("fixed")])

    GeminiDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    assert tools.calls == [["get", "pods", "-n", "arena-x"]]


def test_it_keeps_going_until_the_model_stops_calling_tools():
    tools = RecordingTools()
    client = FakeGemini([
        model_calls(["get", "pods"]),
        model_calls(["describe", "pod", "web-1"]),
        model_calls(["set", "image", "deployment/web", "web=nginx:1.29-alpine"]),
        model_says("I corrected the image tag."),
    ])

    transcript = GeminiDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    assert len(tools.calls) == 3
    assert transcript.stop_reason == "finished"
    assert "corrected the image tag" in transcript.summary


def test_it_handles_several_tool_calls_in_one_turn():
    """Gemini 3 issues parallel calls; dropping the extras would silently lose work."""
    tools = RecordingTools()
    client = FakeGemini([
        model_calls(["get", "pods"], ["get", "events"]),
        model_says("done"),
    ])

    GeminiDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    assert tools.calls == [["get", "pods"], ["get", "events"]]


def test_tool_output_is_fed_back_to_the_model():
    tools = RecordingTools(responses={"get pods": "web-1 0/1 ImagePullBackOff"})
    client = FakeGemini([model_calls(["get", "pods"]), model_says("done")])

    GeminiDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    second_request = client.requests[1]
    assert "ImagePullBackOff" in str(second_request["contents"])


def test_the_models_own_turn_is_replayed_verbatim():
    """Gemini 3 attaches thought signatures to its parts; rebuilding them loses context."""
    client = FakeGemini([model_calls(["get", "pods"]), model_says("done")])

    GeminiDriver(client=client).run(task="fix it", tools=RecordingTools(), budget=Budget())

    history = client.requests[1]["contents"]
    model_turns = [c for c in history if getattr(c, "role", None) == "model"]
    assert model_turns, "the model's own turn must be carried into the next request"


# --- budgets -----------------------------------------------------------------

def test_a_model_that_never_stops_is_cut_off_at_the_budget():
    tools = RecordingTools()
    client = FakeGemini([model_calls(["get", "pods"])] * 50)

    transcript = GeminiDriver(client=client).run(
        task="fix it", tools=tools, budget=Budget(max_turns=4)
    )

    assert len(tools.calls) == 4
    assert transcript.stop_reason == "budget_exhausted"


# --- accounting --------------------------------------------------------------

def test_thinking_tokens_count_towards_output():
    """They are billed as output. Omitting them understates the cost of reasoning."""
    client = FakeGemini([model_says("done")])

    transcript = GeminiDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    # model_says: 10 candidate tokens + 5 thought tokens
    assert transcript.output_tokens == 15
    assert transcript.input_tokens == 50


def test_token_usage_accumulates_across_turns():
    client = FakeGemini([model_calls(["get", "pods"]), model_says("done")])

    transcript = GeminiDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.input_tokens == 150  # 100 + 50
    assert transcript.output_tokens == 65  # (20+30) + (10+5)


def test_the_transcript_records_which_model_produced_it():
    """Preview model ids get retired and repointed; results without one are unciteable."""
    client = FakeGemini([model_says("done")])

    transcript = GeminiDriver(model="gemini-3.6-flash", client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.model == "gemini-3.6-flash"
    assert transcript.driver.startswith("gemini")


# --- failure paths -----------------------------------------------------------

def test_an_api_error_ends_the_run_without_raising():
    """One 500 mid-sweep must not destroy the results collected so far."""
    client = FakeGemini([RuntimeError("503 Service Unavailable")])

    transcript = GeminiDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "error"
    assert "503" in transcript.summary


def test_a_malformed_tool_call_is_recorded_and_the_run_continues():
    """Models emit wrong-shaped arguments; that is a datum, not a crash."""
    from google.genai import types

    bad = types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [
            {"function_call": {"name": "kubectl", "args": {"command": "get pods"}}}
        ]}}],
    })
    tools = RecordingTools()
    client = FakeGemini([bad, model_says("done")])

    transcript = GeminiDriver(client=client).run(
        task="fix it", tools=tools, budget=Budget()
    )

    assert transcript.stop_reason == "finished"
    assert transcript.calls[0].failed


def test_it_refuses_to_start_without_a_key_rather_than_failing_later():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiDriver(api_key="")


# --- applying manifests ------------------------------------------------------
# Some fixes require creating a resource, not editing one. Without a way to feed
# YAML to `kubectl apply -f -`, whole classes of repair are impossible and the
# benchmark would be measuring the harness's limits rather than the agent's.

def test_a_manifest_is_piped_to_kubectl_stdin():
    from google.genai import types

    applying = types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{
            "function_call": {"name": "kubectl", "args": {
                "args": ["apply", "-f", "-"],
                "manifest": "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: x\n",
            }}
        }]}}],
    })
    tools = RecordingTools()
    client = FakeGemini([applying, model_says("done")])

    GeminiDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    assert tools.stdins[0] is not None
    assert "ConfigMap" in tools.stdins[0]


def test_omitting_a_manifest_sends_no_stdin():
    tools = RecordingTools()
    client = FakeGemini([model_calls(["get", "pods"]), model_says("done")])

    GeminiDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    assert tools.stdins == [None]


# --- preflight ---------------------------------------------------------------
# A bad key is only discovered on the first API call, which happens *after* the
# first cluster is provisioned. Across a sweep that is an hour of wall clock to
# learn the credentials were wrong. One cheap call up front turns that into
# three seconds.

def test_preflight_passes_when_the_api_answers():
    driver = GeminiDriver(client=FakeGemini([model_says("ok")]))

    assert driver.preflight() is None


def test_preflight_returns_the_reason_when_the_api_rejects_the_key():
    driver = GeminiDriver(client=FakeGemini([
        RuntimeError("401 API key not valid. Please pass a valid API key.")
    ]))

    problem = driver.preflight()

    assert problem is not None
    assert "401" in problem


def test_preflight_flags_a_key_wrapped_in_smart_quotes():
    """Copying a key out of a document is how this happens, and the shell keeps
    the curly quotes as literal characters."""
    with pytest.raises(ValueError, match="quote"):
        GeminiDriver(api_key="“AIzaSyExampleExampleExampleExampleExa”")


# --- deadlines ---------------------------------------------------------------
# A sweep wedged for 32 minutes on a single hung HTTPS read is what motivated
# these. The SDK will wait forever by default, and Budget.max_seconds existed but
# was never enforced, so one unlucky request stalls every remaining run.

def test_the_http_client_is_given_a_finite_timeout():
    """Without this the SDK blocks on a socket read indefinitely."""
    from arena.drivers.gemini import http_options

    options = http_options(timeout_seconds=90)

    assert options.timeout == 90_000  # the SDK takes milliseconds


class FakeClock:
    def __init__(self, step=0.0):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def test_a_run_that_outlives_its_time_budget_is_stopped():
    tools = RecordingTools()
    client = FakeGemini([model_calls(["get", "pods"])] * 50)
    # Each clock reading advances 30s, so a 60s budget expires almost at once.
    driver = GeminiDriver(client=client, now=FakeClock(step=30.0))

    transcript = driver.run(
        task="fix it", tools=tools, budget=Budget(max_turns=99, max_seconds=60)
    )

    assert transcript.stop_reason == "timeout"
    assert len(tools.calls) < 50


def test_a_run_inside_its_time_budget_is_untouched():
    client = FakeGemini([model_calls(["get", "pods"]), model_says("done")])
    driver = GeminiDriver(client=client, now=FakeClock(step=0.1))

    transcript = driver.run(
        task="fix it", tools=RecordingTools(), budget=Budget(max_seconds=600)
    )

    assert transcript.stop_reason == "finished"
