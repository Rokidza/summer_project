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

LABEL_RULE_VERSION = "1"
"""Bumped whenever the rule below changes what it would call the winner.

Recorded with every set of training examples built from it: a checkpoint
trained under one rule and scored under another is comparing two things.
"""

BASELINE = "none"
"""The passthrough approach: no inference-time technique at all."""

ROUTER = "router"
"""optillm's pretrained router - the thing this platform exists to judge."""

FINETUNED_ROUTER = "router_ft"
"""This repo's finetuned router, served as an optillm plugin.

Its name is the plugin's slug, so a sweep asks for it exactly like any other
approach - see `router_lab/finetuned_router.py` and `plugins/`.
"""

ROUTERS = (ROUTER, FINETUNED_ROUTER)
"""Every router that can be swept, stock first. The order the dashboard offers."""

META_APPROACHES = frozenset(ROUTERS)
"""Approaches that *choose* a technique rather than being one.

They are never eligible to be the actual winner: the label is the target a
router should aim at, so letting a router win its own comparison would be
circular, and would train a future router to defer to another router. That
applies to *our* finetuned router exactly as it does to optillm's: it is trained
on these labels, so leaving it eligible would make the rule circular by the most
direct route available.
"""


@dataclass(frozen=True)
class QueryLabel:
    """The verdict for one query."""

    query_id: str
    winner: str
    any_correct: bool
    """False means no approach was correct and `winner` is the baseline fallback."""
    router_approach: str | None = None
    """What the router being compared picked, when it ran on this query."""
    router: str = ROUTER
    """Which router `router_approach` came from - the comparison's subject."""


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
    eligible = [r for r in results if is_win(r)]
    if not eligible:
        return BASELINE
    return min(eligible, key=_cost_key).approach


def is_win(result: BenchmarkResult) -> bool:
    """Whether a result counts as an approach having answered the query.

    Correct, complete, and not a meta-approach. This module owns the rule, so
    anything that has to know whether an approach succeeded - the label rule
    itself, or the harness deciding which queries stage two still owes work -
    asks here rather than re-deriving it.
    """
    return (
        result.correct
        and result.error is None
        and result.approach not in META_APPROACHES
    )


def label_query(
    results: Sequence[BenchmarkResult], *, router: str = ROUTER
) -> QueryLabel:
    """The full verdict for one query: winner, whether it was a fallback, and
    what `router` predicted (when that router was among the approaches run).

    The winner never depends on `router` - a router's own row is never eligible.
    Which router is *reported* does, so one sweep carrying both of them can be
    read as two comparisons over identical queries rather than one.
    """
    winner = winning_approach(results)
    any_correct = any(is_win(r) for r in results)
    router_approach = next(
        (r.router_approach for r in results if r.approach == router), None
    )
    query_id = results[0].query_id if results else ""
    return QueryLabel(
        query_id=query_id,
        winner=winner,
        any_correct=any_correct,
        router_approach=router_approach,
        router=router,
    )


def label_run(
    store: ResultsStore,
    *,
    run_id: str,
    dataset: str,
    model: str,
    router: str = ROUTER,
) -> list[QueryLabel]:
    """Label every query in one (run, dataset, model) cell."""
    return [
        label_query(
            store.results_for_query(
                run_id=run_id, dataset=dataset, model=model, query_id=query_id
            ),
            router=router,
        )
        for query_id in store.query_ids(run_id=run_id, dataset=dataset, model=model)
    ]
