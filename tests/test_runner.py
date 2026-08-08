"""Surviving a driver that raises.

Only `ClusterError` was caught around a run, so any other exception out of a
driver propagated and ended the sweep - after up to an hour of wall clock, taking
every result collected so far with it. Both API drivers work hard internally to
turn failures into recorded runs rather than exceptions, which only hides how
sharp the edge is: a bug in that handling costs the whole sweep, and an
unexpected response shape is not a rare event across fifteen runs.

A driver that crashes is a failed attempt, not a failed sweep. The cluster is
still there to be graded and snapshotted, so the row keeps its blast radius and
its check results and simply does not pass.

Ctrl-C is the exception to the exception. Someone interrupting a sweep wants out,
not a fifteen-row report of interruptions.
"""

import pytest

from arena.drivers.base import Budget, Transcript
from arena.runner import attempt
from tests.support import RecordingTools


class Working:
    name = "scripted:solution"
    model = ""

    def run(self, task, tools, budget):
        return Transcript(driver=self.name, stop_reason="finished", summary="fixed it")


class Exploding:
    name = "cerebras:gpt-oss-120b"
    model = "gpt-oss-120b"

    def __init__(self, exc):
        self.exc = exc

    def run(self, task, tools, budget):
        raise self.exc


def run(driver):
    return attempt(driver, task="fix it", tools=RecordingTools(), budget=Budget())


def test_a_working_driver_is_passed_through_untouched():
    transcript = run(Working())

    assert transcript.stop_reason == "finished"
    assert transcript.summary == "fixed it"


def test_a_driver_that_raises_does_not_end_the_sweep():
    transcript = run(Exploding(IndexError("list index out of range")))

    assert transcript.stop_reason == "harness_error"


def test_the_crash_is_recorded_where_someone_will_read_it():
    transcript = run(Exploding(IndexError("list index out of range")))

    assert "IndexError" in transcript.summary
    assert "list index out of range" in transcript.summary


def test_the_row_stays_attributable_to_the_driver_that_crashed():
    """A row with no driver or model is not a result anyone can cite."""
    transcript = run(Exploding(RuntimeError("boom")))

    assert transcript.driver == "cerebras:gpt-oss-120b"
    assert transcript.model == "gpt-oss-120b"


def test_no_tool_calls_are_invented_for_a_crashed_run():
    transcript = run(Exploding(RuntimeError("boom")))

    assert transcript.calls == []
    assert transcript.total_tokens == 0


def test_an_interrupt_still_stops_everything():
    """Ctrl-C means stop, not produce fifteen rows saying you pressed Ctrl-C."""
    with pytest.raises(KeyboardInterrupt):
        run(Exploding(KeyboardInterrupt()))


def test_a_system_exit_is_not_swallowed_either():
    with pytest.raises(SystemExit):
        run(Exploding(SystemExit(2)))
