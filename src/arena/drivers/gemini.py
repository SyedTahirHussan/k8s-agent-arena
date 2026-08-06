"""A Gemini-backed operations agent.

The model gets one tool - `kubectl` - and the task text as written. That is a
deliberate choice: giving it a curated set of narrow tools ("scale_deployment",
"fix_image") would smuggle the answer into the tool surface and measure the
harness rather than the agent.

Three details carry more weight than the loop itself:

Thinking tokens are billed as output, so `thoughts_token_count` is added to the
output total. Leaving it out would understate the cost of precisely the models
that reason hardest.

Gemini 3 attaches encrypted thought signatures to its own parts, so each model
turn is appended to history verbatim rather than reconstructed.

Temperature is left unset. Google advises keeping Gemini 3 at its default of 1.0,
and lowering it can induce looping - so run-to-run variation is handled by
repeating runs and reporting the spread, not by pretending sampling is off.
"""

from __future__ import annotations

import os
import re
import time

from arena.drivers.base import Budget, ToolCall, ToolSurface, Transcript

DEFAULT_MODEL = "gemini-3.6-flash"
#: Per-request ceiling. The SDK waits forever by default, and a single hung read
#: otherwise stalls every remaining run in a sweep.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_THINKING_LEVEL = "low"
#: The free tier allows five requests per minute and one agent run needs six to
#: eight, so being throttled mid-run is the normal case, not an edge case.
MAX_RATE_LIMIT_RETRIES = 5
#: Used when the API throttles without saying for how long.
FALLBACK_BACKOFF_SECONDS = 30.0

SYSTEM_INSTRUCTION = """\
You are a Kubernetes operator working on a live cluster. Diagnose the reported \
problem and fix it.

Use the `kubectl` tool to inspect and change the cluster. Pass arguments exactly \
as you would on the command line, as a list of strings: to run `kubectl get pods \
-n web`, pass ["get", "pods", "-n", "web"].

Work only on what the task describes. Do not modify unrelated workloads, and \
prefer the narrowest change that resolves the problem. When you are satisfied the \
problem is fixed, stop calling tools and reply with a short summary of what was \
wrong and what you changed.
"""

KUBECTL_TOOL = {
    "name": "kubectl",
    "description": (
        "Run a kubectl command against the cluster and return its combined "
        "stdout and stderr. Output may be truncated if very long."
    ),
    "parameters_json_schema": {
        "type": "object",
        "properties": {
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "kubectl arguments as separate strings, without the leading "
                    "'kubectl'. Example: ['get', 'pods', '-n', 'default']"
                ),
            },
            "manifest": {
                "type": "string",
                "description": (
                    "Optional YAML piped to the command's stdin. Use it with "
                    "['apply', '-f', '-'] to create or replace a resource."
                ),
            },
        },
        "required": ["args"],
    },
}


def is_daily_quota(exc: Exception) -> bool:
    """Is this the daily cap rather than per-minute throttling?

    The two are worth telling apart. Per-minute throttling clears in seconds; the
    daily cap resets at midnight Pacific, so retrying it burns minutes per run to
    arrive at the same failure.
    """
    return "PerDay" in str(exc)


def is_rate_limited(exc: Exception) -> bool:
    """Is this throttling, as opposed to a request that will fail identically again?"""
    if getattr(exc, "code", None) == 429:
        return True
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def retry_after(exc: Exception, attempt: int) -> float:
    """How long to wait. The API usually says; otherwise back off exponentially.

    A small margin is added to the API's own figure - retrying the instant the
    window opens tends to land just inside it and be refused again.
    """
    if match := re.search(r"retry in ([0-9.]+)s", str(exc)):
        return float(match.group(1)) + 1.0
    return FALLBACK_BACKOFF_SECONDS * (2 ** attempt)


def http_options(timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS):
    """HTTP settings for the client. The SDK expects milliseconds."""
    from google.genai import types

    return types.HttpOptions(timeout=timeout_seconds * 1000)


class GeminiDriver:
    """Drives a Gemini model through a scenario over a kubectl tool surface."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client=None,
        api_key: str | None = None,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
        timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        now=time.monotonic,
        sleep=time.sleep,
    ):
        self.model = model
        self.thinking_level = thinking_level
        self.timeout_seconds = timeout_seconds
        self._now = now
        self._sleep = sleep
        self.name = f"gemini:{model}"

        if client is not None:
            self._client = client
            return

        key = os.environ.get("GEMINI_API_KEY") if api_key is None else api_key
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Export it in your shell or put it in "
                ".env - never pass it on the command line, where it lands in shell "
                "history."
            )

        # Copying a key out of a document or web page brings curly quotes with
        # it, and the shell keeps them as literal characters. The resulting 401
        # is otherwise only discovered after the first cluster is provisioned.
        if any(ch in key for ch in "“”‘’\"'"):
            raise ValueError(
                "GEMINI_API_KEY contains a quote character. This usually means it "
                "was exported with curly quotes copied from a document, e.g. "
                'export GEMINI_API_KEY=“AIza...”, which the shell keeps literally. '
                "Re-export it with straight quotes or none at all."
            )

        from google import genai

        self._client = genai.Client(
            api_key=key, http_options=http_options(self.timeout_seconds)
        )

    def _config(self):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[types.Tool(function_declarations=[KUBECTL_TOOL])],
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            # The harness runs the tools so it can enforce budgets and record
            # every call; the SDK executing them behind our back would leave
            # both unobservable.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def preflight(self) -> str | None:
        """Make one cheap call to prove the credentials work. Returns the problem, or None.

        Without this, a bad key is only discovered on the first API call - which
        happens after the first cluster has been provisioned. Across a sweep that
        is an hour of wall clock spent learning the key was wrong, and because
        API failures are recorded as failed runs rather than raised, the sweep
        completes and every row reads as a model failure.
        """
        from google.genai import types

        try:
            self._client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=[types.Part(text="ok")])],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                    max_output_tokens=16,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - any failure here is disqualifying
            return str(exc)
        return None

    def run(self, task: str, tools: ToolSurface, budget: Budget) -> Transcript:
        from google.genai import types

        contents = [types.Content(role="user", parts=[types.Part(text=task)])]
        calls: list[ToolCall] = []
        input_tokens = output_tokens = 0
        stop_reason = "finished"
        summary = ""
        deadline = self._now() + budget.max_seconds

        while True:
            if self._now() >= deadline:
                stop_reason = "timeout"
                summary = (
                    f"stopped after {len(calls)} tool call(s): exceeded the "
                    f"{budget.max_seconds}s wall-clock budget"
                )
                break

            response = None
            for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    response = self._client.models.generate_content(
                        model=self.model, contents=contents, config=self._config()
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - must not end the sweep
                    if is_daily_quota(exc):
                        stop_reason = "error"
                        summary = (
                            "daily API quota exhausted - this resets at midnight "
                            "Pacific and cannot be waited out. A full sweep needs "
                            "roughly 106 requests; the free tier allows 20 per day. "
                            f"Enable billing to continue. Original error: {exc}"
                        )
                        break
                    if is_rate_limited(exc) and attempt < MAX_RATE_LIMIT_RETRIES:
                        # `contents` is untouched, so the retry resumes the same
                        # conversation rather than restarting the task.
                        self._sleep(retry_after(exc, attempt))
                        continue
                    stop_reason = "error"
                    summary = f"API call failed: {exc}"
                    break

            if response is None:
                break

            if usage := response.usage_metadata:
                input_tokens += usage.prompt_token_count or 0
                # Thinking tokens are billed as output.
                output_tokens += (usage.candidates_token_count or 0) + (
                    usage.thoughts_token_count or 0
                )

            function_calls = response.function_calls
            if not function_calls:
                summary = response.text or ""
                break

            # Verbatim: Gemini 3 carries thought signatures on its own parts.
            if response.candidates:
                contents.append(response.candidates[0].content)

            responses = []
            for call in function_calls:
                if len(calls) >= budget.max_turns:
                    stop_reason = "budget_exhausted"
                    break

                call_args = call.args or {}
                args = call_args.get("args")
                manifest = call_args.get("manifest")
                if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                    output, failed = (
                        f"invalid arguments {call.args!r}: expected {{'args': [str, ...]}}",
                        True,
                    )
                else:
                    output, failed = tools.invoke(args, stdin=manifest)

                calls.append(
                    ToolCall(
                        name="kubectl",
                        arguments=args if isinstance(args, list) else [],
                        result=output,
                        failed=failed,
                    )
                )
                responses.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"output": output, "failed": failed},
                    )
                )

            if responses:
                contents.append(types.Content(role="user", parts=responses))

            if stop_reason == "budget_exhausted":
                summary = f"stopped after {len(calls)} tool call(s): turn budget exhausted"
                break

        return Transcript(
            driver=self.name,
            model=self.model,
            calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            summary=summary,
        )
