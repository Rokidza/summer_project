"""Fixtures and factories shared across the test suite."""

import pytest

from router_lab.labels import FINETUNED_ROUTER, ROUTER
from router_lab.policy import OutcomeTable
from router_lab.store import BenchmarkResult, ResultsStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "results.sqlite"


@pytest.fixture
def store(db_path):
    with ResultsStore.open(db_path) as store:
        yield store


@pytest.fixture
def make_result():
    return build_result


def build_result(**overrides) -> BenchmarkResult:
    """A fully-populated result; override only what a test cares about.

    Every field has a plausible value so a test can state just the one or two
    that matter to it and still get a row the store will accept.
    """
    fields = dict(
        run_id="run-1",
        dataset="gsm8k",
        model="qwen3-8b",
        approach="bon",
        query_id="7",
        question="What is 2 + 2?",
        correct=True,
        predicted="4",
        gold="4",
        latency_s=1.25,
        completion_tokens=128,
        total_tokens=256,
        call_count=3,
        error=None,
        router_approach=None,
        response="The answer is \\boxed{4}",
    )
    fields.update(overrides)
    return BenchmarkResult(**fields)


def query_rows(
    query_id,
    *,
    dataset="gsm8k",
    run_id="run-1",
    model="qwen3-8b",
    question=None,
    router=None,
    router_ft=None,
    **approaches,
):
    """One query's row per approach, given as `name=(correct, total_tokens)`.

    A third element in the tuple is an error message, and `router=(picked,
    correct, tokens)` adds the meta-approach's own row carrying what optillm's
    pretrained router chose - with `router_ft=` the same for the finetuned one, so
    a query swept through both routers is one call. Written this way because
    outcome-based scoring is only ever interesting over a *matrix* of queries by
    approaches, and spelling those out as full results row by row buries the case
    being tested.
    """
    built = [
        build_result(
            run_id=run_id,
            dataset=dataset,
            model=model,
            query_id=query_id,
            question=question or f"question {query_id}",
            approach=approach,
            correct=correct,
            total_tokens=tokens,
            error=error,
        )
        for approach, (correct, tokens, *rest) in approaches.items()
        for error in [rest[0] if rest else None]
    ]
    for approach, decision in ((ROUTER, router), (FINETUNED_ROUTER, router_ft)):
        if decision is None:
            continue
        picked, correct, tokens = decision
        built.append(
            build_result(
                run_id=run_id,
                dataset=dataset,
                model=model,
                query_id=query_id,
                question=question or f"question {query_id}",
                approach=approach,
                correct=correct,
                total_tokens=tokens,
                router_approach=picked,
            )
        )
    return built


def outcome_table(*groups) -> OutcomeTable:
    """An outcome table over several `query_rows` groups."""
    return OutcomeTable.from_results([row for group in groups for row in group])
