# Adding a Cerebras driver

Date: 2026-08-07
Status: approved, not yet implemented

## Why

The benchmark can only measure one agent today, and it cannot finish a sweep.
The Gemini key is on the free tier, which allows 20 requests per day for
`gemini-3.6-flash`. One agent run costs six to eight requests, and a five-scenario
sweep with three repeats needs roughly 106. Exactly one scenario, `configmap-key`,
has ever been measured.

Cerebras serves `gpt-oss-120b` with a far larger free allowance. Measured from
live response headers on 2026-08-07:

| Window | Requests | Tokens    |
| ------ | -------- | --------- |
| minute | 5        | 30,000    |
| hour   | 150      | 1,000,000 |
| day    | 2,400    | 1,000,000 |

The full sweep fits inside five percent of the daily request budget. The binding
constraint moves to the five-per-minute cap, which is what the Gemini free tier
already imposed, so the existing backoff machinery remains necessary rather than
becoming dead weight.

The driver is added rather than substituted. `drivers/base.py` states the reason
in its opening paragraph: identical machinery on identical scenarios is what
makes this a comparison rather than a demo. Replacing Gemini would leave the
benchmark measuring one agent again. Keeping both yields a proprietary-versus-
open-weights axis, which is a more interesting result than either model alone,
and the existing `configmap-key` Gemini row stays valid.

## What already works

No changes are needed to reporting. `Row` is keyed on `(scenario, driver)` and
carries `model`; `RunResult` carries both. Multi-model tables are already
expressible.

Tool calling was verified against the live API before this design was written.
`gpt-oss-120b` returned a well-formed call — `{"args":["-n","web","get","pods"]}` —
with reasoning text in a separate `reasoning` field and `reasoning_tokens` broken
out under `usage.completion_tokens_details`.

## Components

### `src/arena/drivers/cerebras.py`

`CerebrasDriver` follows the shape of `GeminiDriver`: a constructor accepting
`client=None` so tests can inject a fake, a `preflight()` returning the problem
or `None`, and a `run()` returning a `Transcript`.

The client is the `openai` package pointed at Cerebras:

```python
OpenAI(api_key=key, base_url="https://api.cerebras.ai/v1", timeout=120)
```

The `openai` client is chosen over `cerebras-cloud-sdk` because the same driver
then reaches any OpenAI-compatible endpoint — vLLM, Groq, OpenRouter — by
changing one URL. For a benchmark whose purpose is comparing agents, that is
worth more than first-party packaging.

Raw `urllib` is not an option: it is refused by Cloudflare with `error code:
1010`. The SDK sets a User-Agent that is accepted.

The system instruction and the single-`kubectl` tool surface are copied verbatim
from the Gemini driver. An identical prompt is what licenses comparing the two
rows; a reworded one would confound the model difference with a prompt
difference. The tool schema is translated into OpenAI's
`{"type": "function", "function": {...}}` envelope.

Output tokens are read directly from `usage.completion_tokens`, which already
includes reasoning tokens. This differs from Gemini, where `thoughts_token_count`
is a separate field that must be added in.

`--thinking minimal` is rejected with a message naming the accepted values.
`gpt-oss-120b` supports only `low`, `medium` and `high`. Downgrading `minimal` to
`low` silently would record a results row labelled `minimal` for a run that
happened at `low` effort, and a mislabelled measurement is worse than a refused
command.

### Rate limiting

Two separate concerns, and only one of them belongs to the drivers.

**Pacing, in `src/arena/ratelimit.py`.** Both free tiers allow five requests a
minute and one agent run makes six to eight, so a driver that fires as fast as it
can is refused partway through every run. Waiting for the 429 works but costs
more than avoiding it: the API states a wait measured from its own window, which
is longer than the wait a client tracking its own request times would compute,
and a sweep pays that difference fifteen times over.

`RateLimiter` holds a rolling window of request timestamps and sleeps only until
the oldest leaves it. A rolling window rather than an even 12-second gap, because
the burst the provider genuinely allows is worth keeping — a run's opening calls
are the cheap diagnostic ones.

One limiter is shared by every run in a sweep. The allowance belongs to the API
key, and `arena run` builds a fresh driver per repeat, so a limiter owned by a
driver would reset at each scenario and open every one of them with a burst that
gets refused. `cmd_run` constructs it once and passes it into `_make_driver`.

A driver given an injected client — a test double — is not paced by default,
since there is no quota behind it and real sleeps would stall the suite. A driver
that builds its own client always is: unpaced is the wrong default when the
consequence is being throttled on every run.

`--requests-per-minute` overrides the default of 5, so a paid tier is not
throttled by the harness's own free-tier assumption.

**Backoff on 429, per driver.** Pacing reduces how often this path is taken but
cannot replace it: the window is per key, and nothing in this process can see
another process spending the same quota. Retry handling therefore stays inside
each driver rather than moving to a shared module. The loop shape is the same,
but the two providers report the wait differently: Gemini embeds `retry in 4.2s`
in the error text, while Cerebras sends `retry-after` and `x-ratelimit-*`
headers. At two drivers, an abstraction over two dissimilar mechanisms costs more
than the twenty lines it saves.

The Cerebras daily cap of 2,400 requests is distant enough that its
daily-exhaustion path is about producing an honest error message, not about
operational recovery.

### `src/arena/runner.py`

`run_scenario` caught only `ClusterError` around the driver call, so any other
exception a driver raised propagated and ended the sweep — after up to an hour,
taking every result collected so far. Both drivers work hard internally to turn
API failures into recorded runs rather than exceptions, which hides rather than
removes the edge: a bug in that handling costs the whole sweep, and an unexpected
response shape is not a rare event across fifteen runs.

`attempt(driver, task, tools, budget)` runs the driver and converts a crash into
a `harness_error` transcript. The cluster is still standing, so the run keeps its
grading and its blast radius and simply does not pass, and `report.py` already
counts `harness_error` as an incident rather than folding it into a pass rate.
`KeyboardInterrupt` and `SystemExit` are deliberately not caught — someone
interrupting a sweep wants it to stop, not a fifteen-row report of interruptions.

Extracting this as a function is what makes it testable: no test in the suite
provisions a real cluster, so the exception path was unreachable while it lived
inside the `with arena_cluster(...)` block.

### `src/arena/env.py`

The smart-quote guard currently living in `GeminiDriver.__init__` moves here as a
shared function, and both drivers call it. `env.py` already owns credential
reading, so the check belongs beside it. This is not a hypothetical failure: the
Cerebras key was first supplied wrapped in `‘` and `’`, which
`parse_dotenv` would have split into a variable named `'CEREBRAS_API_KEY` holding
a value with a trailing curly quote.

### `src/arena/cli.py`

- `--driver` gains `cerebras`; `_make_driver` gains the corresponding branch.
- `--model` defaults to `None` and is resolved per driver — `gemini-3.6-flash`
  for Gemini, `gpt-oss-120b` for Cerebras. This also corrects existing behaviour:
  the preflight line prints `credentials ok for {args.model}` from the flag
  rather than from the driver, so `--driver noop` currently claims credentials
  for a Gemini model it never contacted.
- Help text for `--model` and `--thinking` no longer names Gemini exclusively.

### `pyproject.toml`

A `cerebras` optional extra holding `openai>=1.0.0`, mirroring the existing
`gemini` extra. The `live_api` marker description names both keys instead of
only `GEMINI_API_KEY`.

## Testing

`tests/test_cerebras.py` mirrors `tests/test_gemini.py`, driving a fake client so
no test spends money or requires a key. Coverage:

- the command the model asks for is the command that runs
- the loop continues until the model stops calling tools
- several tool calls in one turn
- tool output is fed back to the model
- a model that never stops is cut off at the turn budget
- reasoning tokens are counted towards output
- usage accumulates across turns
- the transcript records which model produced it
- an API error ends the run without raising
- a malformed tool call is recorded and the run continues
- a missing key is refused before any cluster is provisioned
- a smart-quoted key is refused with an explanation
- a manifest is piped to kubectl stdin, and its absence sends none
- preflight passes when the API answers, and reports the reason when it does not
- a run that outlives its wall-clock budget is stopped
- a 429 is retried, waited out for as long as the API asked, and eventually
  given up on
- an ordinary error is not retried

## Afterwards

Run `arena run --driver cerebras --repeats 3` across all five scenarios: roughly
106 requests and 160,000 tokens, with a floor of about 21 minutes imposed by the
five-per-minute cap, plus cluster provisioning time.

The results page and README then grow a model dimension. Both must name the
serving provider rather than only the model id, because Cerebras serves
speed-optimised weights and its latency and token figures are not transferable to
the same model served elsewhere.

## Out of scope

Retiring the Gemini driver. Adding further scenarios. Any change to grading,
blast-radius scoring, or cluster provisioning.

## Changes made after the design was approved

Pacing (`ratelimit.py`, `--requests-per-minute`, and the shared-limiter wiring)
was added at the user's direction: reacting to 429s was judged the wrong shape
when the limit is known up front. Both drivers use it, not only the new one.

The `runner.py` crash-handling gap was found during implementation, reported, and
then fixed on the same instruction. It is not specific to the Cerebras driver.
