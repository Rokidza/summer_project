"""Running benchmark sweeps against a served model, recording into the store.

Topology this assumes:

    vLLM    -> holds the weights
    optillm -> proxy in front of vLLM, selects the inference-time approach
    harness -> talks HTTP to optillm and nothing else

The harness never imports optillm. An approach is requested by name in the
request body (`optillm_approach`); the `none` baseline asks for nothing and
passes straight through.
"""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence

from openai import OpenAI

from router_lab.datasets import Problem, category_of
from router_lab.grading import extract_answer, is_correct, system_prompt
from router_lab.labels import BASELINE, is_win
from router_lab.store import (
    SINGLE_STAGE,
    TWO_STAGE,
    BenchmarkResult,
    ResultsStore,
    SweepConfig,
)


@dataclass(frozen=True)
class SweepSettings:
    """How to talk to the endpoint, and how hard to push it."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "sk-no-key"
    concurrency: int = 8
    max_tokens: int = 1536
    temperature: float = 0.6
    timeout: float = 1800.0
    retries: int = 1
    retry_backoff_s: float = 2.0


def new_run_id() -> str:
    """A run id that sorts chronologically, so runs list in the order they ran."""
    return time.strftime("%Y%m%d-%H%M%S")


def _client(settings: SweepSettings) -> OpenAI:
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout,
        max_retries=0,  # retries are the harness's own, so they are recorded
    )


def _usage_of(response) -> tuple[int, int, int]:
    """(completion tokens, total tokens, LLM calls) from a chat completion.

    Two known limits of what optillm reports over HTTP, both of which make cost
    an *under*-estimate rather than a wrong number:

    - Its usage block carries completion tokens only, so total tokens falls
      back to them instead of being recorded as zero. Prompt tokens - which for
      a fan-out approach are paid once per branch - go uncounted.
    - It reports no LLM call count, so a request counts as one call and the
      fan-out shows up in completion tokens alone.

    Both are read from the response when present (`optillm_llm_calls`), so
    these become real numbers the moment optillm reports them, with no change
    here. Until then, treat cost multipliers as a floor.
    """
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or completion_tokens
    calls = getattr(response, "optillm_llm_calls", None) or 1
    return completion_tokens, total_tokens, calls


def solve_one(
    client: OpenAI,
    problem: Problem,
    *,
    run_id: str,
    dataset: str,
    approach: str,
    category: str,
    settings: SweepSettings,
) -> BenchmarkResult:
    """One problem, one approach. Always returns a result - failures are data."""
    extra_body = {} if approach == BASELINE else {"optillm_approach": approach}
    last_error = None

    def record(**outcome) -> BenchmarkResult:
        """Fill in the identity of this (query, approach) cell around an outcome."""
        return BenchmarkResult(
            run_id=run_id,
            dataset=dataset,
            model=settings.model,
            approach=approach,
            query_id=problem.id,
            question=problem.question,
            gold=problem.gold,
            **outcome,
        )

    for attempt in range(settings.retries + 1):
        started = time.time()
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": system_prompt(category)},
                    {"role": "user", "content": problem.question},
                ],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                extra_body=extra_body,
            )
            elapsed = time.time() - started
            text = response.choices[0].message.content or ""
            predicted = extract_answer(text, category)
            completion_tokens, total_tokens, calls = _usage_of(response)
            return record(
                correct=is_correct(predicted, problem.gold, category),
                predicted=predicted,
                latency_s=elapsed,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                call_count=calls,
                error=None,
                router_approach=getattr(response, "optillm_router_approach", None),
                response=text,
            )
        except Exception as exc:  # noqa: BLE001 - any failure is recorded, not raised
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < settings.retries and settings.retry_backoff_s:
                time.sleep(settings.retry_backoff_s * (attempt + 1))

    return record(
        correct=False,
        predicted="",
        latency_s=0.0,
        completion_tokens=0,
        total_tokens=0,
        call_count=0,
        error=last_error,
        router_approach=None,
        response="",
    )


def run_approach(
    store: ResultsStore,
    client: OpenAI,
    problems: Sequence[Problem],
    *,
    run_id: str,
    dataset: str,
    approach: str,
    category: str,
    settings: SweepSettings,
    show_progress: bool = True,
    resume: bool = True,
) -> list[BenchmarkResult]:
    """Run every problem through one approach, writing results as they land.

    Returns every result for this cell - both what it wrote this call and, on
    a resume, what a prior call already wrote - so a caller that must decide
    something from this approach's outcomes (stage one of a two-stage sweep)
    sees the whole picture without reading the store back out itself.

    With `resume` (the default), problems already recorded without an error
    under this exact (run_id, dataset, model, approach) are skipped rather
    than re-run. This is what makes restarting a sweep after a crash, `qdel`,
    or walltime cutoff cheap: reuse the same `--run-id` and only the work that
    never finished gets redone. A row that recorded an error is not treated as
    done, so a resume retries it.
    """
    already: set[str] = set()
    if resume:
        problem_ids = {p.id for p in problems}
        already = store.completed_query_ids(
            run_id=run_id, dataset=dataset, model=settings.model, approach=approach
        ) & problem_ids

    written: list[BenchmarkResult] = []
    if already:
        written.extend(
            r
            for r in store.query_results(
                run_id=run_id, dataset=dataset, approach=approach, model=settings.model
            )
            if r.query_id in already
        )
        if show_progress:
            print(
                f"  [{approach}] resume: {len(already)}/{len(problems)} "
                f"already done, {len(problems) - len(already)} to go"
            )

    todo = [p for p in problems if p.id not in already]
    done = 0
    correct = sum(r.correct for r in written)
    with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
        futures = [
            pool.submit(
                solve_one,
                client,
                problem,
                run_id=run_id,
                dataset=dataset,
                approach=approach,
                category=category,
                settings=settings,
            )
            for problem in todo
        ]
        for future in as_completed(futures):
            result = future.result()
            store.write_result(result)
            written.append(result)
            done += 1
            correct += result.correct
            if show_progress:
                print(
                    f"  [{approach}] {done}/{len(todo)}  "
                    f"running acc {correct / (len(already) + done):.1%}",
                    end="\r",
                    flush=True,
                )
    if show_progress and todo:
        print()
    return written


def run_sweep(
    store: ResultsStore,
    *,
    run_id: str,
    dataset: str,
    approaches: Sequence[str],
    problems: Sequence[Problem],
    settings: SweepSettings,
    show_progress: bool = True,
    resume: bool = True,
) -> None:
    """Run every approach across every problem, recording into the store.

    Results are written per-approach as they complete, so a sweep that dies
    part-way leaves the work it already did behind in the database. With
    `resume` (the default), rerunning this same `run_id` skips whatever
    already completed cleanly instead of redoing it - see `run_approach`.
    """
    category = category_of(dataset)
    client = _client(settings)
    store.record_sweep_config(
        SweepConfig(
            run_id=run_id,
            dataset=dataset,
            model=settings.model,
            mode=SINGLE_STAGE,
            pool_size=len(problems),
            approaches=list(approaches),
        )
    )
    for approach in approaches:
        if show_progress:
            print(f"Running {approach} ...")
        run_approach(
            store,
            client,
            problems,
            run_id=run_id,
            dataset=dataset,
            approach=approach,
            category=category,
            settings=settings,
            show_progress=show_progress,
            resume=resume,
        )


CHEAP_TIER = frozenset({"cot_reflection", "re2", "leap", "z3", "rto"})
"""Single-call approaches whose average cost sits near the baseline's.

A query the baseline answered correctly does *not* mean the baseline was the
cheapest correct option: an audit of the complete-matrix subset found that,
of the queries where the baseline was correct, a cheaper correct alternative
existed 96% of the time - almost always one of these. Restricting them to
`select_stage_two`'s selection (as the expensive tier still is) would mislabel
most of the pool as `none`. They run on every query regardless of stage one's
outcome; only the expensive, fan-out approaches stay restricted to
baseline-wrong-plus-control, since those essentially never beat the baseline's
cost when the baseline was already correct.
"""


def select_stage_two(
    problems: Sequence[Problem],
    baseline_results: Sequence[BenchmarkResult],
    *,
    control_fraction: float,
    seed: int,
) -> list[Problem]:
    """The queries the expensive tier owes work: every baseline failure, plus a
    control.

    This selection is for the *expensive* (fan-out, multi-sample) approaches
    only - see `CHEAP_TIER`'s docstring for why the cheap tier instead runs on
    every query. Restricting the expensive tier is still sound: none of them
    was ever observed to beat the baseline's cost when the baseline was
    already correct, so the label they could only ever contribute there is one
    the cheap tier would already have recorded more cheaply.

    The control sample exists so that, even for the expensive tier, some
    already-solved queries still get run through every approach - without it,
    dataset-level accuracy and cost comparisons for those approaches would
    only ever be drawn from the hard end of the pool.

    What counts as the baseline having answered is `labels.is_win`, so an
    errored result is not a win and its query goes to the expensive tier. The
    sample is drawn in pool order rather than in the order stage one's
    concurrent results happened to land, so the same seed picks the same
    control every time.
    """
    solved = {r.query_id for r in baseline_results if is_win(r)}
    control_pool = [p.id for p in problems if p.id in solved]
    size = min(len(control_pool), math.ceil(control_fraction * len(control_pool)))
    control = set(random.Random(seed).sample(control_pool, size))
    return [p for p in problems if p.id not in solved or p.id in control]


def run_two_stage_sweep(
    store: ResultsStore,
    *,
    run_id: str,
    dataset: str,
    approaches: Sequence[str],
    problems: Sequence[Problem],
    settings: SweepSettings,
    control_fraction: float = 0.1,
    seed: int = 0,
    show_progress: bool = True,
    resume: bool = True,
) -> None:
    """Sweep in two stages, spending the fan-out only where it can change a label.

    Stage one runs the baseline alone across the whole pool. Stage two runs
    the cheap tier (see `CHEAP_TIER`) over every problem - a query the baseline
    solved can still have a cheaper correct winner - and the expensive tier
    over `select_stage_two`'s selection only. The baseline runs whether or not
    it was requested, since both tiers are defined against it.

    Both stages write under one `run_id`, so a two-stage sweep is one run; and
    as with `run_sweep`, results land as they complete, so a sweep that dies
    part-way leaves its finished work behind. With `resume` (the default),
    rerunning this same `run_id` skips whatever already completed cleanly in
    either stage - see `run_approach`.
    """
    category = category_of(dataset)
    client = _client(settings)
    store.record_sweep_config(
        SweepConfig(
            run_id=run_id,
            dataset=dataset,
            model=settings.model,
            mode=TWO_STAGE,
            pool_size=len(problems),
            approaches=list(approaches),
            control_fraction=control_fraction,
            seed=seed,
        )
    )

    if show_progress:
        print(f"Stage 1: {BASELINE} over {len(problems)} problems ...")
    baseline_results = run_approach(
        store,
        client,
        problems,
        run_id=run_id,
        dataset=dataset,
        approach=BASELINE,
        category=category,
        settings=settings,
        show_progress=show_progress,
        resume=resume,
    )

    remaining = [approach for approach in approaches if approach != BASELINE]
    if not remaining:
        return

    cheap = [approach for approach in remaining if approach in CHEAP_TIER]
    expensive = [approach for approach in remaining if approach not in CHEAP_TIER]

    if cheap and show_progress:
        print(f"Stage 2 (cheap tier): {len(problems)}/{len(problems)} problems ({len(cheap)} approaches) ...")
    for approach in cheap:
        if show_progress:
            print(f"Running {approach} ...")
        run_approach(
            store,
            client,
            problems,
            run_id=run_id,
            dataset=dataset,
            approach=approach,
            category=category,
            settings=settings,
            show_progress=show_progress,
            resume=resume,
        )

    if not expensive:
        return

    stage_two = select_stage_two(
        problems,
        baseline_results,
        control_fraction=control_fraction,
        seed=seed,
    )
    if show_progress:
        print(
            f"Stage 2 (expensive tier): {len(stage_two)}/{len(problems)} problems "
            f"({len(expensive)} approaches) ..."
        )
    for approach in expensive:
        if show_progress:
            print(f"Running {approach} ...")
        run_approach(
            store,
            client,
            stage_two,
            run_id=run_id,
            dataset=dataset,
            approach=approach,
            category=category,
            settings=settings,
            show_progress=show_progress,
            resume=resume,
        )
