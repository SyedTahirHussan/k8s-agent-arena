"""Test doubles that are real implementations, not mocks.

`InMemoryCluster` satisfies the same read protocol as the live kubectl-backed
view, so checks under test exercise their actual traversal logic rather than
asserting against a recorded call.
"""

from __future__ import annotations

from typing import Any


def pod(name, namespace="default", labels=None, ready=True, phase="Running",
        restarts=0, **fields):
    conditions = [{"type": "Ready", "status": "True" if ready else "False"}]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace, "labels": labels or {}},
        "status": {
            "phase": phase,
            "conditions": conditions,
            "containerStatuses": [{"name": "main", "restartCount": restarts}],
        },
        **fields,
    }


def deployment(name, namespace="default", image="nginx:1.29", replicas=1, **fields):
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "replicas": replicas,
            "template": {
                "spec": {"containers": [{"name": "web", "image": image}]}
            },
        },
        **fields,
    }


def model_calls(*commands: list[str]):
    """A Gemini response that asks to run one or more kubectl commands."""
    from google.genai import types

    return types.GenerateContentResponse.model_validate({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"function_call": {"name": "kubectl", "args": {"args": command}}}
                    for command in commands
                ],
            }
        }],
        "usage_metadata": {
            "prompt_token_count": 100,
            "candidates_token_count": 20,
            "thoughts_token_count": 30,
            "total_token_count": 150,
        },
    })


def model_says(text: str):
    """A Gemini response that stops and reports, calling no tools."""
    from google.genai import types

    return types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}],
        "usage_metadata": {
            "prompt_token_count": 50,
            "candidates_token_count": 10,
            "thoughts_token_count": 5,
            "total_token_count": 65,
        },
    })


class FakeGemini:
    """A stand-in for `genai.Client` that replays prepared responses.

    Built from the SDK's real response types, so the driver exercises genuine
    parsing rather than an invented shape. Raises whatever exception is queued,
    which is how the API-failure paths get covered.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if not self.responses:
            return model_says("done")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _completion(message: dict, usage: dict, finish_reason: str):
    """Build a real `ChatCompletion` so the driver parses genuine SDK objects."""
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-oss-120b",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": usage,
    })


def cerebras_calls(*commands: list[str], reasoning: str = "checking the pods"):
    """A Cerebras response that asks to run one or more kubectl commands.

    Token counts mirror `model_calls` so the two drivers' accounting tests read
    the same, but the mechanism differs: `completion_tokens` already contains
    `reasoning_tokens`, where Gemini reports thinking separately.
    """
    import json

    return _completion(
        {
            "role": "assistant",
            "reasoning": reasoning,
            "tool_calls": [
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {
                        "name": "kubectl",
                        "arguments": json.dumps({"args": command}),
                    },
                }
                for index, command in enumerate(commands)
            ],
        },
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "completion_tokens_details": {"reasoning_tokens": 30}},
        "tool_calls",
    )


def cerebras_applies(command: list[str], manifest: str):
    """A response that pipes a manifest to kubectl's stdin."""
    import json

    return _completion(
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-0",
                "type": "function",
                "function": {
                    "name": "kubectl",
                    "arguments": json.dumps({"args": command, "manifest": manifest}),
                },
            }],
        },
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "completion_tokens_details": {"reasoning_tokens": 30}},
        "tool_calls",
    )


def cerebras_says(text: str):
    """A Cerebras response that stops and reports, calling no tools."""
    return _completion(
        {"role": "assistant", "content": text},
        {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65,
         "completion_tokens_details": {"reasoning_tokens": 5}},
        "stop",
    )


class FakeCerebras:
    """A stand-in for `openai.OpenAI` that replays prepared responses.

    Shaped like the real client - `client.chat.completions.create(...)` - and
    hands back real `ChatCompletion` objects, so the driver does genuine parsing.
    Queued exceptions are raised, which is how the failure paths get covered.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            return cerebras_says("done")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def rate_limited(retry_after: str | None = None, remaining_day: str | None = None):
    """A genuine `openai.RateLimitError`, built from a real 429 response.

    Cerebras states the wait in headers rather than in the message text, so the
    headers are what the driver has to read.
    """
    import httpx
    import openai

    headers = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    if remaining_day is not None:
        headers["x-ratelimit-remaining-requests-day"] = remaining_day

    request = httpx.Request("POST", "https://api.cerebras.ai/v1/chat/completions")
    response = httpx.Response(
        429, headers=headers, request=request,
        json={"error": {"message": "rate limit exceeded"}},
    )
    return openai.RateLimitError("rate limit exceeded", response=response, body=None)


class RecordingTools:
    """A kubectl surface that records invocations instead of running them.

    Real enough to drive a driver: it returns canned output, can be told which
    commands should fail, and keeps the call log the tests assert against.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        failures: set[str] | None = None,
    ):
        self.responses = responses or {}
        self.failures = failures or set()
        self.calls: list[list[str]] = []
        self.stdins: list[str | None] = []

    def invoke(self, args: list[str], stdin: str | None = None) -> tuple[str, bool]:
        """Run a kubectl command. Returns (output, failed)."""
        self.calls.append(list(args))
        self.stdins.append(stdin)
        key = " ".join(args)
        if key in self.failures:
            return (f"error: unknown command {key!r}", True)
        return (self.responses.get(key, ""), False)


class InMemoryCluster:
    """A cluster view backed by a list of resource dicts."""

    def __init__(self, items: list[dict[str, Any]] | None = None):
        self.items = list(items or [])

    def get(self, kind: str, namespace: str | None, name: str) -> dict[str, Any] | None:
        for item in self.items:
            metadata = item.get("metadata", {})
            if (
                item.get("kind") == kind
                and metadata.get("namespace") == namespace
                and metadata.get("name") == name
            ):
                return item
        return None

    def list(
        self, kind: str, namespace: str | None, selector: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        matches = []
        for item in self.items:
            metadata = item.get("metadata", {})
            if item.get("kind") != kind or metadata.get("namespace") != namespace:
                continue
            labels = metadata.get("labels", {})
            if selector and any(labels.get(k) != v for k, v in selector.items()):
                continue
            matches.append(item)
        return matches
