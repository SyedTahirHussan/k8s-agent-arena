"""A Cerebras-backed operations agent.

Same loop as the Gemini driver, same task text, same single `kubectl` tool. That
sameness is the point: a reworded prompt or a different tool surface would
confound the model difference with a harness difference, and the comparison
between a proprietary model and an open-weights one would stop meaning anything.

Three details differ from Gemini and each one can quietly corrupt a result.

`completion_tokens` already contains `reasoning_tokens`, where Gemini reports
thinking in a separate field that has to be added in. Adding it again here would
inflate this model's cost against the other.

The wait after a 429 arrives in a `retry-after` header rather than in the error
text, and `x-ratelimit-remaining-requests-day` is what distinguishes an exhausted
daily allowance from ordinary per-minute throttling.

The OpenAI SDK retries failed requests on its own unless told not to. Those
retries are invisible in the transcript but still spend wall clock, which the
results would then attribute to the model rather than to the API.
"""

from __future__ import annotations

import json
import os
import time

from arena.drivers.base import Budget, ToolCall, ToolSurface, Transcript
from arena.env import require_key
from arena.ratelimit import RateLimiter

DEFAULT_MODEL = "gpt-oss-120b"
BASE_URL = "https://api.cerebras.ai/v1"
#: Per-request ceiling. Without one a single hung read stalls every remaining run
#: in a sweep.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_REASONING_EFFORT = "low"
#: gpt-oss-120b accepts these and nothing else. `minimal`, which Gemini supports,
#: is deliberately absent.
REASONING_EFFORTS = ("low", "medium", "high")
#: How many times one request is retried before the run is given up on.
#: Throttling and 5xx are both worth retrying; a provider that is still
#: refusing after five attempts is down, and waiting longer wedges the sweep.
MAX_TRANSIENT_RETRIES = 5
#: Used when a 429 arrives without saying how long to wait.
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
    "type": "function",
    "function": {
        "name": "kubectl",
        "description": (
            "Run a kubectl command against the cluster and return its combined "
            "stdout and stderr. Output may be truncated if very long."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "kubectl arguments as separate strings, without the "
                        "leading 'kubectl'. Example: ['get', 'pods', '-n', 'default']"
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
    },
}


def response_headers(exc: Exception) -> dict:
    """The headers off a failed request, or nothing if this was not an HTTP error."""
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None) or {}


def is_daily_quota(exc: Exception) -> bool:
    """Is the daily allowance gone, as opposed to per-minute throttling?

    Worth telling apart. Per-minute throttling clears in seconds; the daily
    allowance refills on its own schedule, so sleeping through five retries only
    burns minutes to arrive at the same failure.
    """
    return response_headers(exc).get("x-ratelimit-remaining-requests-day") == "0"


def is_rate_limited(exc: Exception) -> bool:
    """Is this throttling, as opposed to a request that will fail identically again?"""
    if getattr(exc, "status_code", None) == 429:
        return True
    return "429" in str(exc)


def is_transient(exc: Exception) -> bool:
    """Is this worth trying again?

    Throttling and 5xx both are: the provider is busy or having a bad second, and
    the next request usually works. A sweep of fifteen runs met two 500s and lost
    a whole run to each - the model issued no tool calls at all, so the row held
    no information about it while still filling a slot in the table.

    A 4xx is not. It is our request that is wrong, and it will be refused
    identically however many times it is sent.
    """
    status = getattr(exc, "status_code", None)
    return is_rate_limited(exc) or (status is not None and 500 <= status < 600)


def retry_after(exc: Exception, attempt: int) -> float:
    """How long to wait. Cerebras usually says; otherwise back off exponentially.

    A small margin is added to the stated figure - retrying the instant the window
    opens tends to land just inside it and be refused again.
    """
    stated = response_headers(exc).get("retry-after")
    if stated:
        try:
            return float(stated) + 1.0
        except ValueError:
            # A date-formatted `retry-after` is legal HTTP and not worth parsing;
            # backing off is the safe reading.
            pass
    return FALLBACK_BACKOFF_SECONDS * (2 ** attempt)


def build_client(api_key: str, timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS):
    """An OpenAI client pointed at Cerebras, with its own retries turned off.

    The generic client rather than `cerebras-cloud-sdk` because the same driver
    then reaches any OpenAI-compatible endpoint by changing one URL, which is
    what a benchmark built to compare agents wants.
    """
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=timeout_seconds,
        # The harness does the waiting, so that every second spent on backoff is
        # visible in the transcript rather than hidden inside the SDK.
        max_retries=0,
    )


class CerebrasDriver:
    """Drives a Cerebras-hosted model through a scenario over a kubectl surface."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client=None,
        api_key: str | None = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        limiter: RateLimiter | None = None,
        now=time.monotonic,
        sleep=time.sleep,
    ):
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning effort {reasoning_effort!r} is not supported by "
                f"{model}, which accepts {', '.join(REASONING_EFFORTS)}. "
                "Mapping 'minimal' onto 'low' would label a result with an effort "
                "the model never ran at."
            )

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self._now = now
        self._sleep = sleep
        self.name = f"cerebras:{model}"

        if client is not None:
            self._client = client
            # An injected client is a test double with no quota behind it. Pacing
            # one would spend real seconds protecting nothing.
            self.limiter = limiter
            return

        key = require_key(
            "CEREBRAS_API_KEY",
            os.environ.get("CEREBRAS_API_KEY") if api_key is None else api_key,
        )
        self._client = build_client(key, self.timeout_seconds)
        # A sweep shares one limiter across its runs, because the allowance
        # belongs to the key. A driver built on its own still needs a safe
        # default: unpaced, it is throttled partway through every run.
        self.limiter = limiter or RateLimiter()

    def _create(self, messages: list, **overrides):
        # Every attempt counts against the window, retries included.
        if self.limiter is not None:
            self.limiter.acquire()

        request = {
            "model": self.model,
            "messages": messages,
            "reasoning_effort": self.reasoning_effort,
        }
        request.update(overrides)
        return self._client.chat.completions.create(**request)

    def preflight(self) -> str | None:
        """Make one cheap call to prove the credentials work. Returns the problem, or None.

        Without this, a bad key is only discovered on the first API call - which
        happens after the first cluster has been provisioned. Across a sweep that
        is an hour of wall clock spent learning the key was wrong, and because
        API failures are recorded as failed runs rather than raised, the sweep
        completes and every row reads as a model failure.
        """
        try:
            self._create(
                [{"role": "user", "content": "ok"}], max_completion_tokens=16
            )
        except Exception as exc:  # noqa: BLE001 - any failure here is disqualifying
            return str(exc)
        return None

    def run(self, task: str, tools: ToolSurface, budget: Budget) -> Transcript:
        messages: list = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": task},
        ]
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
            for attempt in range(MAX_TRANSIENT_RETRIES + 1):
                try:
                    response = self._create(messages, tools=[KUBECTL_TOOL])
                    break
                except Exception as exc:  # noqa: BLE001 - must not end the sweep
                    if is_daily_quota(exc):
                        stop_reason = "error"
                        summary = (
                            "daily API request allowance exhausted - this refills "
                            "on its own schedule and cannot be waited out within a "
                            "run. A full sweep needs roughly 106 requests. "
                            f"Original error: {exc}"
                        )
                        break
                    if is_transient(exc) and attempt < MAX_TRANSIENT_RETRIES:
                        # `messages` is untouched, so the retry resumes the same
                        # conversation rather than restarting the task.
                        self._sleep(retry_after(exc, attempt))
                        continue
                    stop_reason = "error"
                    summary = f"API call failed: {exc}"
                    break

            if response is None:
                break

            if usage := response.usage:
                input_tokens += usage.prompt_tokens or 0
                # Reasoning tokens are already inside `completion_tokens`.
                output_tokens += usage.completion_tokens or 0

            # Only ClusterError is caught around a run, so raising here would end
            # the sweep and lose every result collected so far.
            if not response.choices:
                stop_reason = "error"
                summary = "API returned no choices to act on"
                break

            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            if not tool_calls:
                summary = message.content or ""
                break

            # Verbatim: gpt-oss carries its reasoning on its own turn, and
            # rebuilding the turn loses that context mid-task.
            messages.append(message.model_dump(exclude_none=True))

            for call in tool_calls:
                if len(calls) >= budget.max_turns:
                    stop_reason = "budget_exhausted"
                    break

                args, manifest, problem = _parse_arguments(call.function.arguments)
                if problem:
                    output, failed = problem, True
                else:
                    output, failed = tools.invoke(args, stdin=manifest)

                calls.append(
                    ToolCall(
                        name="kubectl",
                        arguments=args,
                        result=output,
                        failed=failed,
                    )
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps({"output": output, "failed": failed}),
                })

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


def _parse_arguments(raw: str) -> tuple[list[str], str | None, str]:
    """Pull `args` and `manifest` out of a tool call. Returns the problem, or "".

    A model emitting wrong-shaped arguments is a datum about the model, and
    truncated output leaves invalid JSON. Neither should end a sweep, so both come
    back as a failed tool call the model can see and correct.
    """
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return [], None, f"invalid tool arguments, not valid JSON: {raw!r}"

    if not isinstance(payload, dict):
        return [], None, f"invalid tool arguments {payload!r}: expected an object"

    args = payload.get("args")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return [], None, (
            f"invalid arguments {payload!r}: expected {{'args': [str, ...]}}"
        )

    manifest = payload.get("manifest")
    return args, manifest if isinstance(manifest, str) else None, ""
