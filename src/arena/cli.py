"""Command line entry point.

    arena list                          # what scenarios exist
    arena run --driver solution         # prove the scenarios are solvable
    arena run --driver noop             # establish the floor
    arena run --driver gemini --repeats 3

Results stream to stdout as each run lands, because a sweep takes minutes per
scenario and a crash on the last one should not lose the rest.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from arena.cluster import delete_cluster, list_clusters, orphaned_clusters
from arena.env import load_dotenv
from arena.drivers.base import Budget
from arena.drivers.noop import NoopDriver
from arena.drivers.scripted import ScriptedDriver
from arena.report import aggregate, to_markdown, write_results
from arena.runner import RunResult, run_scenario
from arena.scenario import Scenario, ScenarioError, load_scenario

#: Scenarios live beside the source in a checkout. Installed as a wheel they do
#: not ship, so fall back to the working directory.
SCENARIO_ROOT = Path(__file__).resolve().parents[2] / "scenarios"


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root if (root / "scenarios").is_dir() else Path.cwd()


def _scenario_root() -> Path:
    return SCENARIO_ROOT if SCENARIO_ROOT.is_dir() else Path.cwd() / "scenarios"


def discover(root: Path | None = None) -> list[Scenario]:
    """Load every scenario, loudly rejecting any that will not parse."""
    root = root or _scenario_root()
    if not root.is_dir():
        raise SystemExit(f"no scenarios directory found at {root}")
    return [load_scenario(path) for path in sorted(root.glob("*/scenario.yaml"))]


def _make_driver(kind: str, scenario: Scenario, model: str, thinking: str):
    if kind == "noop":
        return NoopDriver()
    if kind == "solution":
        if not scenario.solution:
            raise SystemExit(
                f"scenario {scenario.id!r} has no reference solution to replay"
            )
        return ScriptedDriver(scenario.solution, name="solution")
    if kind == "gemini":
        from arena.drivers.gemini import GeminiDriver

        try:
            return GeminiDriver(model=model, thinking_level=thinking)
        except ValueError as exc:
            # A missing key is a setup mistake, not a bug. Say so plainly rather
            # than printing a traceback.
            raise SystemExit(str(exc)) from exc
    raise SystemExit(f"unknown driver {kind!r}")


def _print_result(result: RunResult) -> None:
    mark = "PASS" if result.passed else "FAIL"
    print(
        f"  [{mark}] {result.scenario:22} "
        f"checks {result.checks_passed}/{result.checks_total}  "
        f"blast {result.blast_score}  "
        f"calls {result.tool_calls}  "
        f"tokens {result.total_tokens:,}  "
        f"{result.duration_seconds}s",
        flush=True,
    )
    if result.error:
        print(f"         harness error: {result.error}", flush=True)
    elif not result.passed:
        for line in result.verification.splitlines():
            print(f"         {line}", flush=True)
    if result.collateral:
        print(f"         collateral: {', '.join(result.collateral)}", flush=True)


def cmd_list(args: argparse.Namespace) -> int:
    for scenario in discover():
        solvable = "yes" if scenario.solution else "no reference solution"
        print(f"{scenario.id:22} {scenario.title}")
        print(f"{'':22} namespace={scenario.namespace} solution={solvable}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        scenarios = discover()
    except ScenarioError as exc:
        print(f"invalid scenario: {exc}", file=sys.stderr)
        return 1
    print(f"{len(scenarios)} scenario(s) parsed cleanly")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Delete clusters an interrupted run left behind."""
    orphans = orphaned_clusters(list_clusters())
    if not orphans:
        print("no leftover arena clusters")
        return 0

    for name in orphans:
        print(f"deleting {name} ...", flush=True)
        delete_cluster(name)
    print(f"removed {len(orphans)} cluster(s)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    scenarios = discover()
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.id in wanted]
        missing = wanted - {s.id for s in scenarios}
        if missing:
            print(f"no such scenario(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 1

    budget = Budget(max_turns=args.max_turns)
    results: list[RunResult] = []

    # Prove the credentials work before provisioning anything. API failures are
    # recorded as failed runs rather than raised, so without this a bad key
    # produces a complete sweep in which every row looks like a model failure.
    probe = _make_driver(args.driver, scenarios[0], args.model, args.thinking)
    if check := getattr(probe, "preflight", None):
        if problem := check():
            print(f"preflight failed, not starting the sweep:\n  {problem}", file=sys.stderr)
            return 1
        print(f"credentials ok for {args.model}")

    print(f"driver={args.driver} repeats={args.repeats} scenarios={len(scenarios)}\n")
    for scenario in scenarios:
        print(f"{scenario.id}: {scenario.title}", flush=True)
        for attempt in range(args.repeats):
            driver = _make_driver(args.driver, scenario, args.model, args.thinking)
            result = run_scenario(scenario, driver, budget=budget, keep=args.keep)
            results.append(result)
            _print_result(result)

    rows = aggregate(results)
    print("\n" + to_markdown(rows))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else Path("results") / f"{args.driver}-{stamp}.json"
    write_results(results, out)
    print(f"\nraw results: {out}")

    # A sweep that produced no passes is worth a non-zero exit in CI.
    return 0 if any(r.passed for r in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arena", description="Benchmark LLM agents operating Kubernetes clusters"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available scenarios").set_defaults(func=cmd_list)
    sub.add_parser("validate", help="check every scenario parses").set_defaults(
        func=cmd_validate
    )
    sub.add_parser(
        "clean", help="delete arena clusters left behind by an interrupted run"
    ).set_defaults(func=cmd_clean)

    run = sub.add_parser("run", help="run scenarios against a driver")
    run.add_argument(
        "--driver", default="solution",
        choices=["noop", "solution", "gemini"],
        help="agent under test (default: solution, which replays the known fix)",
    )
    run.add_argument("--scenario", action="append", help="scenario id (repeatable)")
    run.add_argument("--repeats", type=int, default=1,
                     help="runs per scenario; >1 is how variance gets reported")
    run.add_argument("--model", default="gemini-3.6-flash", help="model id for --driver gemini")
    run.add_argument("--thinking", default="low",
                     choices=["minimal", "low", "medium", "high"],
                     help="Gemini thinking level")
    run.add_argument("--max-turns", type=int, default=25, help="tool-call budget per run")
    run.add_argument("--keep", action="store_true",
                     help="leave clusters running for post-mortem (you must delete them)")
    run.add_argument("--out", help="where to write raw JSON results")
    run.set_defaults(func=cmd_run)

    # Read .env before dispatching, so the documented setup actually works.
    load_dotenv(_project_root() / ".env")

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print(
            "\ninterrupted. If a cluster was mid-creation, `arena clean` removes "
            "anything left behind.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
