"""Aggregating repeats into a table.

Gemini 3 is meant to be run at its default temperature, so the same agent on the
same scenario genuinely does not give the same answer twice. A single run is an
anecdote. The report therefore never collapses repeats into a boolean - a
scenario passed 2 of 3 times is reported as 2/3, because that is what happened.
"""

from arena.report import aggregate, to_markdown
from arena.runner import RunResult


def result(scenario="imagepull-backoff", driver="gemini:gemini-3.6-flash",
           passed=True, blast=0, tokens=(1000, 200), calls=4):
    return RunResult(
        scenario=scenario,
        title=scenario.replace("-", " ").title(),
        driver=driver,
        model="gemini-3.6-flash",
        passed=passed,
        checks_passed=2 if passed else 0,
        checks_total=2,
        verification="",
        blast_score=blast,
        blast_summary="",
        tool_calls=calls,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
    )


def test_repeats_are_reported_as_k_of_n_not_as_a_boolean():
    rows = aggregate([result(passed=True), result(passed=False), result(passed=True)])

    (row,) = rows
    assert row.passes == 2
    assert row.runs == 3
    assert row.pass_rate == "2/3"


def test_a_scenario_that_never_passed_is_reported_as_zero_of_n():
    rows = aggregate([result(passed=False), result(passed=False)])

    assert rows[0].pass_rate == "0/2"


def test_results_are_grouped_by_scenario_and_driver():
    rows = aggregate([
        result(scenario="a", driver="gemini:flash"),
        result(scenario="a", driver="noop"),
        result(scenario="b", driver="gemini:flash"),
    ])

    assert len(rows) == 3


def test_the_same_scenario_under_two_drivers_does_not_get_merged():
    """Merging the control into the model's score would erase the comparison."""
    rows = aggregate([
        result(driver="gemini:flash", passed=True),
        result(driver="noop", passed=False),
    ])

    by_driver = {row.driver: row.pass_rate for row in rows}
    assert by_driver == {"gemini:flash": "1/1", "noop": "0/1"}


def test_blast_radius_is_averaged_across_repeats():
    rows = aggregate([result(blast=0), result(blast=6)])

    assert rows[0].mean_blast == 3.0


def test_the_worst_blast_radius_is_kept_not_just_the_mean():
    """One destructive run in five matters, and a mean of 1.2 hides it."""
    rows = aggregate([result(blast=0)] * 4 + [result(blast=30)])

    assert rows[0].max_blast == 30


def test_token_cost_is_totalled():
    rows = aggregate([result(tokens=(1000, 200)), result(tokens=(2000, 400))])

    assert rows[0].total_tokens == 3600


def test_an_empty_run_set_aggregates_to_nothing_rather_than_raising():
    assert aggregate([]) == []


# --- rendering ---------------------------------------------------------------

def test_the_table_shows_the_pass_rate_and_the_blast_radius():
    table = to_markdown(aggregate([result(passed=True, blast=3)]))

    assert "1/1" in table
    assert "imagepull-backoff" in table


def test_the_table_survives_having_no_results():
    assert isinstance(to_markdown([]), str)
