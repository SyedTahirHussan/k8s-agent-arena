"""Reading credentials from a .env file.

The setup instructions say to copy `.env.example` to `.env`, so `.env` has to be
read or the documented path silently does nothing and presents as a missing key.

Deliberately tiny, and deliberately not a dependency. The only rule that carries
any weight: a real environment variable always wins. Someone exporting a key for
a single command expects that to take effect rather than be overridden by a file
they set up weeks ago and forgot.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse `KEY=value` lines, tolerating comments, blanks and `export` prefixes."""
    settings: dict[str, str] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        # People paste the line straight out of their shell.
        line = line.removeprefix("export ").lstrip()

        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not name:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        settings[name] = value

    return settings


#: Both curly and straight. A key never legitimately contains one, so any of them
#: is a paste artefact rather than part of the credential.
_QUOTES = "“”‘’\"'"


def require_key(name: str, value: str | None) -> str:
    """Return ``value``, or explain what is wrong with it.

    A bad key is otherwise only discovered on the first API call, which happens
    after the first cluster has been provisioned - an hour of wall clock across a
    sweep to learn the credentials were never going to work.
    """
    if not value:
        raise ValueError(
            f"{name} is not set. Export it in your shell or put it in .env - "
            "never pass it on the command line, where it lands in shell history."
        )

    # Copying a key out of a document or web page brings curly quotes with it,
    # and the shell keeps them as literal characters.
    if any(ch in value for ch in _QUOTES):
        raise ValueError(
            f"{name} contains a quote character. This usually means it was "
            f"exported with quotes copied from a document, e.g. {name}=“sk-...”, "
            "which the shell keeps literally. Re-export it with no quotes at all."
        )

    return value


def load_dotenv(path: str | Path = ".env") -> None:
    """Load ``path`` into the environment without overriding anything already set."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError:
        # No .env is the normal case, not a problem worth reporting.
        return

    for name, value in parse_dotenv(text).items():
        os.environ.setdefault(name, value)
