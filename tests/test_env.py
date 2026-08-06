"""Loading credentials from a .env file.

The README tells you to copy `.env.example` to `.env`, so `.env` has to actually
be read - otherwise the documented setup silently does nothing and the failure
looks like a missing key.

A real environment variable always wins. Someone who exports a key for one
command expects that to take effect, not to be quietly overridden by a stale
file they set up weeks ago.
"""

from arena.env import load_dotenv, parse_dotenv


def test_it_reads_a_simple_assignment():
    assert parse_dotenv("GEMINI_API_KEY=abc123") == {"GEMINI_API_KEY": "abc123"}


def test_it_ignores_comments_and_blank_lines():
    text = """
    # a comment

    GEMINI_API_KEY=abc123
    """
    assert parse_dotenv(text) == {"GEMINI_API_KEY": "abc123"}


def test_it_strips_wrapping_quotes():
    assert parse_dotenv('GEMINI_API_KEY="abc123"') == {"GEMINI_API_KEY": "abc123"}
    assert parse_dotenv("GEMINI_API_KEY='abc123'") == {"GEMINI_API_KEY": "abc123"}


def test_it_keeps_an_export_prefix_from_leaking_into_the_name():
    """People paste the line straight out of their shell."""
    assert parse_dotenv("export GEMINI_API_KEY=abc123") == {"GEMINI_API_KEY": "abc123"}


def test_it_ignores_a_line_with_no_assignment():
    assert parse_dotenv("this is not a setting") == {}


def test_it_keeps_equals_signs_inside_the_value():
    assert parse_dotenv("TOKEN=a=b=c") == {"TOKEN": "a=b=c"}


def test_a_real_environment_variable_wins_over_the_file(tmp_path, monkeypatch):
    """An explicit export for one command must not be overridden by a stale file."""
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-file")
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")

    load_dotenv(env_file)

    import os
    assert os.environ["GEMINI_API_KEY"] == "from-shell"


def test_the_file_fills_in_what_the_shell_did_not_set(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-file")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    load_dotenv(env_file)

    import os
    assert os.environ["GEMINI_API_KEY"] == "from-file"


def test_a_missing_file_is_not_an_error(tmp_path):
    load_dotenv(tmp_path / "nope.env")
