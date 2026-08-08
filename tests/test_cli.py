"""Wiring the command line to a driver.

Two drivers now answer to `arena run`, and they do not share a default model.
A single `--model` default belonging to one of them is how a sweep ends up
labelled with a model it never called - the preflight line already printed
`credentials ok for gemini-3.6-flash` under `--driver noop`, which contacts
nothing at all.

Setup mistakes are reported as setup mistakes. A tracebacks-on-typos command line
teaches people to ignore its output.
"""

import pytest

from arena.cli import _make_driver, main, resolve_model

#: Only the `solution` driver reads the scenario, so passing nothing here is the
#: point: an API-backed driver that needed one would be reaching for the answer.
NO_SCENARIO = None


# --- per-driver model defaults -----------------------------------------------

def test_cerebras_defaults_to_its_own_model():
    assert resolve_model("cerebras", None) == "gpt-oss-120b"


def test_gemini_defaults_to_its_own_model():
    assert resolve_model("gemini", None) == "gemini-3.6-flash"


def test_an_explicit_model_wins_over_the_default():
    assert resolve_model("cerebras", "zai-glm-4.7") == "zai-glm-4.7"


def test_a_driver_that_calls_no_model_reports_none():
    """`noop` and `solution` contact no API, so naming a model would be a lie."""
    assert resolve_model("noop", None) is None
    assert resolve_model("solution", None) is None


# --- building drivers --------------------------------------------------------

def test_the_cerebras_driver_is_built_with_the_resolved_model(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-example")

    driver = _make_driver("cerebras", NO_SCENARIO, "gpt-oss-120b", "low")

    assert driver.model == "gpt-oss-120b"
    assert driver.name == "cerebras:gpt-oss-120b"


def test_a_missing_key_is_a_setup_error_not_a_traceback(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="CEREBRAS_API_KEY"):
        _make_driver("cerebras", NO_SCENARIO, "gpt-oss-120b", "low")


def test_an_unsupported_thinking_level_is_a_setup_error(monkeypatch):
    """gpt-oss-120b has no `minimal`, and silently downgrading mislabels the run."""
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-example")

    with pytest.raises(SystemExit, match="minimal"):
        _make_driver("cerebras", NO_SCENARIO, "gpt-oss-120b", "minimal")


def test_an_unknown_driver_is_rejected():
    with pytest.raises(SystemExit, match="unknown driver"):
        _make_driver("gpt5", NO_SCENARIO, "whatever", "low")


# --- the parser --------------------------------------------------------------

def test_cerebras_is_offered_on_the_command_line(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--driver", "nonsense"])

    assert "cerebras" in capsys.readouterr().err


# --- pacing ------------------------------------------------------------------
# The allowance belongs to the API key, not the run, and `arena run` builds a
# fresh driver per repeat. A limiter owned by the driver would therefore reset at
# every scenario and open each one with a burst the provider refuses.

def test_a_driver_is_given_the_sweeps_limiter(monkeypatch):
    from arena.ratelimit import RateLimiter

    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-example")
    limiter = RateLimiter(max_requests=3)

    driver = _make_driver("cerebras", NO_SCENARIO, "gpt-oss-120b", "low", limiter)

    assert driver.limiter is limiter


def test_gemini_is_paced_by_the_same_seam(monkeypatch):
    from arena.ratelimit import RateLimiter

    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyExample")
    limiter = RateLimiter(max_requests=3)

    driver = _make_driver("gemini", NO_SCENARIO, "gemini-3.6-flash", "low", limiter)

    assert driver.limiter is limiter


def test_the_requests_per_minute_default_matches_the_free_tier():
    from arena.cli import build_parser

    args = build_parser().parse_args(["run", "--driver", "cerebras"])

    assert args.requests_per_minute == 5


def test_a_paid_tier_can_raise_the_pacing():
    from arena.cli import build_parser

    args = build_parser().parse_args(
        ["run", "--driver", "cerebras", "--requests-per-minute", "60"]
    )

    assert args.requests_per_minute == 60
