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
) -> list[BenchmarkResult]:
    """Run every problem through one approach, writing results as they land.

    Returns what it wrote, so a caller that must decide something from this
    approach's outcomes - stage one of a two-stage sweep - does not have to read
    the rows back out of the store.
    """
    done = 0
    correct = 0
    written: list[BenchmarkResult] = []
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
            for problem in problems
        ]
        for future in as_completed(futures):
            result = future.result()
            store.write_result(result)
            written.append(result)
            done += 1
            correct += result.correct
            if show_progress:
                print(
                    f"  [{approach}] {done}/{len(problems)}  "
                    f"running acc {correct / done:.1%}",
                    end="\r",
                    flush=True,
                )
    if show_progress:
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
) -> None:
    """Run every approach across every problem, recording into the store.

    Results are written per-approach as they complete, so a sweep that dies
    part-way leaves the work it already did behind in the database.
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
        )


def select_stage_two(
    problems: Sequence[Problem],
    baseline_results: Sequence[BenchmarkResult],
    *,
    control_fraction: float,
    seed: int,
) -> list[Problem]:
    """The queries stage two owes work: every baseline failure, plus a control.

    A query the baseline answered correctly already has its label - the label
    rule makes the baseline the automatic winner there - so running twelve more
    approaches on it buys a known outcome. The control sample is the exception:
    without some of the solved queries also being run through every approach,
    dataset-level accuracy and cost comparisons would only ever be drawn from
    the hard end of the pool.

    What counts as the baseline having answered is `labels.is_win`, so an
    errored result is not a win and its query goes to stage two. The sample is
    drawn in pool order rather than in the order stage one's concurrent results
    happened to land, so the same seed picks the same control every time.
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
) -> None:
    """Sweep in two stages, spending the fan-out only where it can change a label.

    Stage one runs the baseline alone across the whole pool. Stage two runs
    every other approach over `select_stage_two`'s selection. The baseline runs
    whether or not it was requested, since stage two is defined against it.

    Both stages write under one `run_id`, so a two-stage sweep is one run; and
    as with `run_sweep`, results land as they complete, so a sweep that dies
    part-way leaves its finished work behind.
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
    )

    remaining = [approach for approach in approaches if approach != BASELINE]
    if not remaining:
        return

    stage_two = select_stage_two(
        problems,
        baseline_results,
        control_fraction=control_fraction,
        seed=seed,
    )
    if show_progress:
        print(
            f"Stage 2: {len(stage_two)}/{len(problems)} problems "
            f"({len(remaining)} approaches) ..."
        )
    for approach in remaining:
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
        )
