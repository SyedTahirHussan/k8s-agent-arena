"""The driver seam and the two drivers that need no API key.

A driver is anything that can be handed a task and a kubectl surface and left to
work. Keeping that seam thin is what lets Gemini, K8sGPT and a do-nothing control
be scored by identical machinery - the comparison is the point of the benchmark,
not any single agent.

`NoopDriver` is the control: whatever score it gets is what a scenario awards for
doing nothing, and a scenario it passes is a broken scenario. `ScriptedDriver`
replays fixed commands, which is how the harness itself gets tested without
spending money or inheriting a model's nondeterminism.
"""

import pytest

from arena.drivers.base import Budget, Transcript
from arena.drivers.noop import NoopDriver
from arena.drivers.scripted import ScriptedDriver
from tests.support import RecordingTools


# --- the control condition ---------------------------------------------------

def test_the_noop_driver_touches_nothing():
    tools = RecordingTools()

    transcript = NoopDriver().run(task="fix the cluster", tools=tools, budget=Budget())

    assert tools.calls == []
    assert transcript.calls == []


def test_the_noop_driver_still_returns_a_scorable_transcript():
    """The control has to flow through the same reporting path as a real agent."""
    transcript = NoopDriver().run(task="fix it", tools=RecordingTools(), budget=Budget())

    assert isinstance(transcript, Transcript)
    assert transcript.driver == "noop"
    assert transcript.total_tokens == 0


# --- replaying fixed commands ------------------------------------------------

def test_a_scripted_driver_runs_its_commands_in_order():
    tools = RecordingTools()
    driver = ScriptedDriver([
        ["get", "pods"],
        ["set", "image", "deployment/web", "web=nginx:1.29-alpine"],
    ])

    driver.run(task="fix it", tools=tools, budget=Budget())

    assert tools.calls == [
        ["get", "pods"],
        ["set", "image", "deployment/web", "web=nginx:1.29-alpine"],
    ]


def test_a_scripted_driver_records_what_each_command_returned():
    tools = RecordingTools(responses={"get pods": "web-1  0/1  ImagePullBackOff"})
    driver = ScriptedDriver([["get", "pods"]])

    transcript = driver.run(task="fix it", tools=tools, budget=Budget())

    assert transcript.calls[0].arguments == ["get", "pods"]
    assert "ImagePullBackOff" in transcript.calls[0].result


def test_a_driver_stops_when_it_runs_out_of_turns():
    """Budgets are enforced by the harness, not trusted to the agent."""
    tools = RecordingTools()
    driver = ScriptedDriver([["get", "pods"]] * 10)

    transcript = driver.run(task="fix it", tools=tools, budget=Budget(max_turns=3))

    assert len(tools.calls) == 3
    assert transcript.stop_reason == "budget_exhausted"


def test_a_driver_that_finishes_its_script_reports_completion():
    transcript = ScriptedDriver([["get", "pods"]]).run(
        task="fix it", tools=RecordingTools(), budget=Budget()
    )

    assert transcript.stop_reason == "finished"


def test_a_failing_command_does_not_abort_the_run():
    """Agents issue invalid kubectl constantly; that is data, not a crash."""
    tools = RecordingTools(failures={"delete nonsense"})
    driver = ScriptedDriver([["delete", "nonsense"], ["get", "pods"]])

    transcript = driver.run(task="fix it", tools=tools, budget=Budget())

    assert len(transcript.calls) == 2
    assert transcript.calls[0].failed
    assert not transcript.calls[1].failed


# --- budget ------------------------------------------------------------------

def test_budgets_have_defaults_so_a_runaway_agent_always_stops():
    budget = Budget()

    assert budget.max_turns > 0
    assert budget.max_seconds > 0


@pytest.mark.parametrize("bad", [0, -1])
def test_a_nonsensical_turn_budget_is_rejected(bad):
    with pytest.raises(ValueError):
        Budget(max_turns=bad)


# --- transcript accounting ---------------------------------------------------

def test_a_transcript_totals_its_token_usage():
    transcript = Transcript(driver="x", calls=[], input_tokens=1200, output_tokens=340)

    assert transcript.total_tokens == 1540


def test_a_transcript_counts_how_many_commands_failed():
    tools = RecordingTools(failures={"delete nonsense"})
    transcript = ScriptedDriver([["delete", "nonsense"], ["get", "pods"]]).run(
        task="fix it", tools=tools, budget=Budget()
    )

    assert transcript.failed_calls == 1
