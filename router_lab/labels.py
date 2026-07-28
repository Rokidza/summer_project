"""Which approach actually won - the ground-truth label for router training.

The rule: among the approaches that answered a query correctly, the winner is
the cheapest one; if none answered correctly, the winner is the baseline. Cost
is the raw token/call proxy recorded per result - never a dollar figure.

`winning_approach` and `label_query` are pure: callers hand them results they
already fetched. Only `label_run` touches the store, and only through its query
surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from router_lab.store import BenchmarkResult, ResultsStore

BASELINE = "none"
"""The passthrough approach: no inference-time technique at all."""

ROUTER = "router"
"""optillm's pretrained router - the thing this platform exists to judge."""

META_APPROACHES = frozenset({ROUTER})
"""Approaches that *choose* a technique rather than being one.

They are never eligible to be the actual winner: the label is the target a
router should aim at, so letting a router win its own comparison would be
circular, and would train a future router to defer to another router.
"""


@dataclass(frozen=True)
class QueryLabel:
    """The verdict for one query."""

    query_id: str
    winner: str
    any_correct: bool
    """False means no approach was correct and `winner` is the baseline fallback."""
    router_approach: str | None = None
    """What optillm's pretrained router picked, when the `router` approach ran."""


def _cost_key(result: BenchmarkResult) -> tuple:
    """Order results cheapest-first, deterministically.

    Tokens are the primary proxy; call count and latency break ties, and the
    approach name breaks the remainder so the rule never depends on input order.
    """
    return (
        result.total_tokens,
        result.call_count,
        result.latency_s,
        result.approach,
    )


def winning_approach(results: Iterable[BenchmarkResult]) -> str:
    """The cheapest approach that answered correctly, else the baseline.

    Results that errored are never eligible, even if flagged correct - a result
    that failed to complete has no trustworthy cost or answer. Neither are the
    meta-approaches, which route rather than reason.
    """
    eligible = [r for r in results if _is_candidate(r)]
    if not eligible:
        return BASELINE
    return min(eligible, key=_cost_key).approach


def _is_candidate(result: BenchmarkResult) -> bool:
    return (
        result.correct
        and result.error is None
        and result.approach not in META_APPROACHES
    )


def label_query(results: Sequence[BenchmarkResult]) -> QueryLabel:
    """The full verdict for one query: winner, whether it was a fallback, and
    what optillm's router predicted (when the `router` approach was among them)."""
    winner = winning_approach(results)
    any_correct = any(_is_candidate(r) for r in results)
    router_approach = next(
        (r.router_approach for r in results if r.approach == ROUTER), None
    )
    query_id = results[0].query_id if results else ""
    return QueryLabel(
        query_id=query_id,
        winner=winner,
        any_correct=any_correct,
        router_approach=router_approach,
    )


def label_run(
    store: ResultsStore, *, run_id: str, dataset: str, model: str
) -> list[QueryLabel]:
    """Label every query in one (run, dataset, model) cell."""
    return [
        label_query(
            store.results_for_query(
                run_id=run_id, dataset=dataset, model=model, query_id=query_id
            )
        )
        for query_id in store.query_ids(run_id=run_id, dataset=dataset, model=model)
    ]
