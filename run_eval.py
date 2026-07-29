#!/usr/bin/env python3
"""Run a benchmark sweep against a served model, recording into the SQLite store.

Topology:
    vLLM      -> http://localhost:8001/v1   (holds the weights)
    optillm   -> http://localhost:8000/v1   (proxy, --base-url points at vLLM)

This script only talks to optillm over HTTP. It never imports optillm.

Results go into one queryable database rather than per-run JSONL/CSV files;
browse them with the dashboard:

    streamlit run router_lab/dashboard.py -- --db results.sqlite

Usage:
    python run_eval.py --model qwen3-8b --approaches none bon
    python run_eval.py --model qwen3-8b --approaches none bon moa router --limit 50
    python run_eval.py --model qwen3-8b --approaches none bon moa --two-stage

A full sweep is the platform's dominant expense, and most of it is spent on
queries the baseline already answers correctly - which the label rule hands to
the baseline regardless. `--two-stage` runs the baseline over the whole pool
first, then the remaining approaches over what it got wrong plus a seeded
control sample of what it got right.
"""

import argparse
import os

from router_lab.datasets import DATASETS, load_problems
from router_lab.harness import (
    SweepSettings,
    new_run_id,
    run_sweep,
    run_two_stage_sweep,
)
from router_lab.store import DB_ENV_VAR, DEFAULT_DB, ResultsStore
from router_lab.views import format_leaderboard, leaderboard


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="vLLM --served-model-name")
    ap.add_argument(
        "--approaches",
        nargs="+",
        default=["none", "bon"],
        help="'none' is the baseline passthrough",
    )
    ap.add_argument("--dataset", default="gsm8k", choices=sorted(DATASETS))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument(
        "--db",
        default=os.environ.get(DB_ENV_VAR, DEFAULT_DB),
        help=f"results database path (default: ${DB_ENV_VAR} or {DEFAULT_DB})",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="defaults to a timestamp; reuse one to extend an existing run",
    )
    ap.add_argument(
        "--two-stage",
        action="store_true",
        help="baseline first, then the rest only where the baseline failed",
    )
    ap.add_argument(
        "--control-fraction",
        type=float,
        default=0.1,
        help="two-stage: share of baseline-solved queries stage two also runs",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="two-stage: makes the control sample reproducible",
    )
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "sk-no-key"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--retries", type=int, default=1)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_id = args.run_id or new_run_id()

    print(f"Loading {args.dataset} ...")
    problems = load_problems(args.dataset, args.limit)
    mode = "two-stage" if args.two_stage else "single-stage"
    print(
        f"run {run_id} ({mode}): {len(problems)} problems, "
        f"approaches: {', '.join(args.approaches)}\n"
    )

    settings = SweepSettings(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
    )

    with ResultsStore.open(args.db) as store:
        if args.two_stage:
            run_two_stage_sweep(
                store,
                run_id=run_id,
                dataset=args.dataset,
                approaches=args.approaches,
                problems=problems,
                settings=settings,
                control_fraction=args.control_fraction,
                seed=args.seed,
            )
        else:
            run_sweep(
                store,
                run_id=run_id,
                dataset=args.dataset,
                approaches=args.approaches,
                problems=problems,
                settings=settings,
            )
        rows = leaderboard(
            store, run_id=run_id, dataset=args.dataset, model=args.model
        )

    print("\n" + format_leaderboard(rows) + "\n")
    print(f"results in {args.db} (run {run_id})")


if __name__ == "__main__":
    main()
