"""The Cerebras driver.

The same agent loop as the Gemini driver, against an OpenAI-compatible API, and
the differences are exactly where a benchmark can quietly lie.

Reasoning tokens are already inside `completion_tokens` here, where Gemini
reports them separately. Adding them again would inflate the open-weights model's
cost against the proprietary one - a difference in the comparison that came from
the harness rather than the models.

`reasoning_effort` accepts only low, medium and high. `minimal` is refused rather
than quietly downgraded, because a row labelled `minimal` for a run that happened
at `low` is a wrong measurement, which is worse than a refused command.

And the SDK retries 429s by itself unless told not to. Hidden retries would be
invisible in the transcript while still spending wall clock the results attribute
to the model.
"""

import pytest

from arena.drivers.base import Budget
from arena.drivers.cerebras import CerebrasDriver
from tests.support import (
    FakeCerebras,
    RecordingTools,
    cerebras_applies,
    cerebras_calls,
    cerebras_says,
    rate_limited,
)


def test_it_runs_the_command_the_model_asks_for():
    tools = RecordingTools()
    client = FakeCerebras([cerebras_calls(["get", "pods", "-n", "arena-x"]),
                           cerebras_says("fixed")])

    CerebrasDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    assert tools.calls == [["get", "pods", "-n", "arena-x"]]


def test_it_keeps_going_until_the_model_stops_calling_tools():
    tools = RecordingTools()
    client = FakeCerebras([
        cerebras_calls(["get", "pods"]),
        cerebras_calls(["describe", "pod", "web-1"]),
        cerebras_calls(["set", "image", "deployment/web", "web=nginx:1.29-alpine"]),
        cerebras_says("I corrected the image tag."),
    ])

    transcript = CerebrasDriver(client=client).run(
        task="fix it", tools=tools, budget=Budget()
    )

    assert len(tools.calls) == 3
    assert transcript.stop_reason == "finished"
    assert "corrected the image tag" in transcript.summary


def test_it_handles_several_tool_calls_in_one_turn():
    tools = RecordingTools()
    client = FakeCerebras([
        cerebras_calls(["get", "pods"], ["get", "events"]),
        cerebras_says("done"),
    ])

    CerebrasDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    assert tools.calls == [["get", "pods"], ["get", "events"]]


def test_tool_output_is_fed_back_to_the_model():
    tools = RecordingTools(responses={"get pods": "web-1 0/1 ImagePullBackOff"})
    client = FakeCerebras([cerebras_calls(["get", "pods"]), cerebras_says("done")])

    CerebrasDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    assert "ImagePullBackOff" in str(client.requests[1]["messages"])


def test_a_tool_result_is_tied_to_the_call_it_answers():
    """Without the id the API cannot match result to call and rejects the turn."""
    client = FakeCerebras([cerebras_calls(["get", "pods"]), cerebras_says("done")])

    CerebrasDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    tool_messages = [
        message for message in client.requests[1]["messages"]
        if _role(message) == "tool"
    ]
    assert [_field(message, "tool_call_id") for message in tool_messages] == ["call-0"]


def test_the_models_own_turn_is_replayed_verbatim():
    """gpt-oss carries its reasoning on its own turn; rebuilding it loses context."""
    client = FakeCerebras([cerebras_calls(["get", "pods"]), cerebras_says("done")])

    CerebrasDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    history = client.requests[1]["messages"]
    assistant_turns = [m for m in history if _role(m) == "assistant"]
    assert assistant_turns, "the model's own turn must be carried into the next request"
    assert "checking the pods" in str(assistant_turns[0])


def _role(message):
    return message.get("role") if isinstance(message, dict) else getattr(message, "role", None)


def _field(message, name):
    return message.get(name) if isinstance(message, dict) else getattr(message, name, None)


# --- budgets -----------------------------------------------------------------

def test_a_model_that_never_stops_is_cut_off_at_the_budget():
    tools = RecordingTools()
    client = FakeCerebras([cerebras_calls(["get", "pods"])] * 50)

    transcript = CerebrasDriver(client=client).run(
        task="fix it", tools=tools, budget=Budget(max_turns=4)
    )

    assert len(tools.calls) == 4
    assert transcript.stop_reason == "budget_exhausted"


# --- accounting --------------------------------------------------------------

def test_reasoning_tokens_are_not_counted_twice():
    """They are already inside `completion_tokens`, unlike Gemini's separate field."""
    client = FakeCerebras([cerebras_says("done")])

    transcript = CerebrasDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    # cerebras_says: 15 completion tokens, of which 5 are reasoning.
    assert transcript.output_tokens == 15
    assert transcript.input_tokens == 50


def test_token_usage_accumulates_across_turns():
    client = FakeCerebras([cerebras_calls(["get", "pods"]), cerebras_says("done")])

    transcript = CerebrasDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.input_tokens == 150  # 100 + 50
    assert transcript.output_tokens == 65  # 50 + 15


def test_the_transcript_records_which_model_produced_it():
    client = FakeCerebras([cerebras_says("done")])

    transcript = CerebrasDriver(model="gpt-oss-120b", client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.model == "gpt-oss-120b"
    assert transcript.driver.startswith("cerebras")


# --- failure paths -----------------------------------------------------------

def test_an_api_error_ends_the_run_without_raising():
    """One failure mid-sweep must not destroy the results collected so far."""
    client = FakeCerebras([RuntimeError("400 Bad Request")])

    transcript = CerebrasDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "error"
    assert "400" in transcript.summary


def test_a_malformed_tool_call_is_recorded_and_the_run_continues():
    """Models emit wrong-shaped arguments; that is a datum, not a crash."""
    from tests.support import _completion

    bad = _completion(
        {"role": "assistant", "tool_calls": [{
            "id": "call-0", "type": "function",
            "function": {"name": "kubectl", "arguments": '{"command": "get pods"}'},
        }]},
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "tool_calls",
    )
    client = FakeCerebras([bad, cerebras_says("done")])

    transcript = CerebrasDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "finished"
    assert transcript.calls[0].failed


def test_unparseable_tool_arguments_are_recorded_rather_than_raising():
    """Truncated output leaves invalid JSON, and json.loads would end the sweep."""
    from tests.support import _completion

    truncated = _completion(
        {"role": "assistant", "tool_calls": [{
            "id": "call-0", "type": "function",
            "function": {"name": "kubectl", "arguments": '{"args": ["get", "pod'},
        }]},
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "tool_calls",
    )
    client = FakeCerebras([truncated, cerebras_says("done")])

    transcript = CerebrasDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "finished"
    assert transcript.calls[0].failed


def test_it_refuses_to_start_without_a_key_rather_than_failing_later():
    with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
        CerebrasDriver(api_key="")


def test_it_refuses_a_key_wrapped_in_smart_quotes():
    """Copying a key out of a document is how this happens, and the shell keeps
    the curly quotes as literal characters."""
    with pytest.raises(ValueError, match="quote"):
        CerebrasDriver(api_key="‘csk-exampleexampleexampleexample’")


# --- reasoning effort --------------------------------------------------------

def test_the_requested_reasoning_effort_reaches_the_api():
    client = FakeCerebras([cerebras_says("done")])

    CerebrasDriver(client=client, reasoning_effort="high").run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert client.requests[0]["reasoning_effort"] == "high"


def test_minimal_effort_is_refused_rather_than_quietly_downgraded():
    """A row labelled `minimal` for a run that happened at `low` is a wrong result."""
    with pytest.raises(ValueError, match="minimal"):
        CerebrasDriver(client=FakeCerebras([]), reasoning_effort="minimal")


# --- applying manifests ------------------------------------------------------

def test_a_manifest_is_piped_to_kubectl_stdin():
    tools = RecordingTools()
    client = FakeCerebras([
        cerebras_applies(
            ["apply", "-f", "-"],
            "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: x\n",
        ),
        cerebras_says("done"),
    ])

    CerebrasDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    assert tools.stdins[0] is not None
    assert "ConfigMap" in tools.stdins[0]


def test_omitting_a_manifest_sends_no_stdin():
    tools = RecordingTools()
    client = FakeCerebras([cerebras_calls(["get", "pods"]), cerebras_says("done")])

    CerebrasDriver(client=client).run(task="fix it", tools=tools, budget=Budget())

    assert tools.stdins == [None]


# --- preflight ---------------------------------------------------------------

def test_preflight_passes_when_the_api_answers():
    driver = CerebrasDriver(client=FakeCerebras([cerebras_says("ok")]))

    assert driver.preflight() is None


def test_preflight_returns_the_reason_when_the_api_rejects_the_key():
    driver = CerebrasDriver(client=FakeCerebras([
        RuntimeError("401 Wrong API Key provided")
    ]))

    problem = driver.preflight()

    assert problem is not None
    assert "401" in problem


# --- deadlines ---------------------------------------------------------------

class FakeClock:
    def __init__(self, step=0.0):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def test_a_run_that_outlives_its_time_budget_is_stopped():
    tools = RecordingTools()
    client = FakeCerebras([cerebras_calls(["get", "pods"])] * 50)
    driver = CerebrasDriver(client=client, now=FakeClock(step=30.0))

    transcript = driver.run(
        task="fix it", tools=tools, budget=Budget(max_turns=99, max_seconds=60)
    )

    assert transcript.stop_reason == "timeout"
    assert len(tools.calls) < 50


def test_a_run_inside_its_time_budget_is_untouched():
    client = FakeCerebras([cerebras_calls(["get", "pods"]), cerebras_says("done")])
    driver = CerebrasDriver(client=client, now=FakeClock(step=0.1))

    transcript = driver.run(
        task="fix it", tools=RecordingTools(), budget=Budget(max_seconds=600)
    )

    assert transcript.stop_reason == "finished"


# --- rate limits -------------------------------------------------------------
# Cerebras allows five requests per minute and one agent run makes six to eight,
# so being throttled mid-run is the normal case. Unlike Gemini it states the wait
# in a `retry-after` header rather than in the error text.

class Naps:
    def __init__(self):
        self.slept = []

    def __call__(self, seconds):
        self.slept.append(seconds)


def test_a_rate_limited_request_is_retried_rather_than_failing_the_run():
    naps = Naps()
    client = FakeCerebras([rate_limited(retry_after="20"), cerebras_says("done")])

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "finished"
    assert naps.slept, "it should have waited before retrying"


def test_it_waits_as_long_as_the_header_asked():
    naps = Naps()
    client = FakeCerebras([rate_limited(retry_after="20"), cerebras_says("done")])

    CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    # 20s requested; a small margin on top is fine, undershooting is not.
    assert naps.slept[0] >= 20


def test_a_429_without_a_header_still_backs_off():
    naps = Naps()
    client = FakeCerebras([rate_limited(), cerebras_says("done")])

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "finished"
    assert naps.slept[0] > 0


def test_it_gives_up_after_repeated_rate_limits():
    naps = Naps()
    client = FakeCerebras([rate_limited(retry_after="1")] * 20)

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "error"
    assert len(naps.slept) <= 6, "it must not retry forever"


def test_an_ordinary_error_is_not_retried():
    """Only throttling is worth waiting out; a 400 will fail again identically."""
    naps = Naps()
    client = FakeCerebras([RuntimeError("400 invalid request"), cerebras_says("done")])

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "error"
    assert naps.slept == []


def test_retrying_does_not_lose_the_conversation():
    """A retry must resend the same history, not restart the task."""
    naps = Naps()
    client = FakeCerebras([
        cerebras_calls(["get", "pods"]),
        rate_limited(retry_after="1"),
        cerebras_says("done"),
    ])

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "finished"
    assert len(transcript.calls) == 1
    assert "get" in str(client.requests[-1]["messages"])


# --- the daily cap is not worth waiting out ----------------------------------
# 2,400 requests a day is far enough away that this is about an honest error
# message rather than operational recovery - but a sweep that hits it should say
# so instead of sleeping through five pointless retries.

def test_an_exhausted_daily_cap_is_not_retried():
    naps = Naps()
    client = FakeCerebras([rate_limited(retry_after="30", remaining_day="0")] * 10)

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "error"
    assert naps.slept == [], "the daily cap resets on its own schedule"
    assert "daily" in transcript.summary.lower()


def test_per_minute_throttling_is_still_waited_out():
    """The distinction has to be precise or the useful retry is lost too."""
    naps = Naps()
    client = FakeCerebras([
        rate_limited(retry_after="12", remaining_day="2399"),
        cerebras_says("done"),
    ])

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "finished"
    assert naps.slept


# --- the SDK's own retries ---------------------------------------------------

def test_the_sdk_does_not_retry_behind_the_harnesss_back():
    """Hidden retries spend wall clock the results would attribute to the model."""
    from arena.drivers.cerebras import build_client

    client = build_client(api_key="csk-example", timeout_seconds=90)

    assert client.max_retries == 0
    assert client.timeout == 90


def test_the_client_is_pointed_at_cerebras():
    from arena.drivers.cerebras import build_client

    client = build_client(api_key="csk-example", timeout_seconds=90)

    assert "cerebras.ai" in str(client.base_url)


# --- responses that are not shaped like responses ----------------------------
# Only ClusterError is caught around a run, so anything a driver raises ends the
# whole sweep and takes every result collected so far with it. A response missing
# the parts the driver reads is a bad API day, not a reason to lose 40 minutes of
# measurements.

def test_a_response_with_no_choices_ends_the_run_without_raising():
    from tests.support import _completion

    empty = _completion(
        {"role": "assistant", "content": "hi"},
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop",
    )
    empty.choices = []
    client = FakeCerebras([empty])

    transcript = CerebrasDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "error"
    assert "choices" in transcript.summary.lower()


# --- pacing ------------------------------------------------------------------
# Five requests a minute against six-to-eight per run means a 429 partway through
# every run unless the requests are spaced up front. Waiting for the refusal
# works but costs more: the API's stated wait is measured from its window, not
# from ours.

class PacingClock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_requests_are_paced_to_the_providers_window():
    from arena.ratelimit import RateLimiter

    clock = PacingClock()
    limiter = RateLimiter(max_requests=2, per_seconds=60.0, now=clock, sleep=clock.sleep)
    client = FakeCerebras([
        cerebras_calls(["get", "pods"]),
        cerebras_calls(["get", "events"]),
        cerebras_says("done"),
    ])

    CerebrasDriver(client=client, limiter=limiter, now=clock).run(
        task="fix it", tools=RecordingTools(), budget=Budget(max_seconds=10_000)
    )

    assert clock.slept, "the third request had to wait for the window"
    assert len(client.requests) == 3


def test_a_retry_after_throttling_is_also_paced():
    """A retry is another request against the same window, not a free pass."""
    from arena.ratelimit import RateLimiter

    clock = PacingClock()
    limiter = RateLimiter(max_requests=1, per_seconds=60.0, now=clock, sleep=clock.sleep)
    client = FakeCerebras([rate_limited(retry_after="1"), cerebras_says("done")])

    CerebrasDriver(client=client, limiter=limiter, now=clock, sleep=lambda s: None).run(
        task="fix it", tools=RecordingTools(), budget=Budget(max_seconds=10_000)
    )

    assert clock.slept, "the retry is a second request and must be paced too"


def test_an_injected_client_is_not_paced_by_default():
    """A test double has no quota to protect, and real sleeps would stall the suite."""
    client = FakeCerebras([cerebras_calls(["get", "pods"])] * 30)

    transcript = CerebrasDriver(client=client).run(
        task="fix it", tools=RecordingTools(), budget=Budget(max_turns=20)
    )

    assert transcript.stop_reason == "budget_exhausted"


def test_a_real_client_is_paced_by_default(monkeypatch):
    """The default has to be safe; an unpaced sweep is throttled on every run."""
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-example")

    driver = CerebrasDriver()

    assert driver.limiter is not None
    assert driver.limiter.max_requests == 5


# --- transient server errors -------------------------------------------------
# A sweep of fifteen runs met two HTTP 500s, and each one cost a whole run: the
# model never issued a single tool call, so the row carried no information about
# it while still occupying a slot in the results table. A 500 is the provider
# having a bad second, and the request that follows it usually works.

def a_server_error(status=500):
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.cerebras.ai/v1/chat/completions")
    response = httpx.Response(
        status, request=request,
        json={"detail": "An unexpected error occurred"},
    )
    return openai.InternalServerError(
        f"Error code: {status}", response=response, body=None
    )


def test_a_transient_server_error_is_retried():
    naps = Naps()
    client = FakeCerebras([a_server_error(), cerebras_says("done")])

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "finished"
    assert naps.slept, "it should have waited before retrying"


def test_a_bad_gateway_is_retried_too():
    naps = Naps()
    client = FakeCerebras([a_server_error(502), cerebras_says("done")])

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "finished"


def test_repeated_server_errors_still_give_up():
    """A provider that is down stays down; retrying forever wedges the sweep."""
    naps = Naps()
    client = FakeCerebras([a_server_error()] * 20)

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "error"
    assert len(naps.slept) <= 6


def test_a_client_error_is_still_not_retried():
    """A 400 is our mistake and will be refused identically every time."""
    naps = Naps()
    client = FakeCerebras([a_server_error(400), cerebras_says("done")])

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "error"
    assert naps.slept == []


def test_retrying_a_server_error_does_not_lose_the_conversation():
    naps = Naps()
    client = FakeCerebras([
        cerebras_calls(["get", "pods"]),
        a_server_error(),
        cerebras_says("done"),
    ])

    transcript = CerebrasDriver(client=client, sleep=naps).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "finished"
    assert len(transcript.calls) == 1
    assert "get" in str(client.requests[-1]["messages"])
