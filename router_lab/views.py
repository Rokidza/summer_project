"""Shaped, ready-to-render answers about a results database.

This layer sits between the store and everything that displays results - the
console summary the harness prints, and the Streamlit dashboard. It writes no
SQL and holds no schema knowledge: it composes `router_lab.store`'s query
surface with the label rule from `router_lab.labels`.

Cost is always a multiplier over the `none` baseline's tokens, never a price.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from router_lab.datasets import DATASETS, category_of
from router_lab.labels import BASELINE, ROUTER, ROUTERS, label_query, label_run
from router_lab.store import ResultsStore


# -- leaderboard ----------------------------------------------------------


@dataclass(frozen=True)
class LeaderboardRow:
    """One approach's standing on one (dataset, model)."""

    approach: str
    dataset: str
    model: str
    n: int
    correct: int
    accuracy: float
    delta_pp: float | None
    """Accuracy minus the baseline's, in percentage points; None without a baseline."""
    errors: int
    avg_latency_s: float
    avg_completion_tokens: float
    total_tokens: int
    cost_multiplier: float | None
    """Total tokens as a multiple of the baseline's; None without a baseline."""


def leaderboard(
    store: ResultsStore,
    *,
    run_id: str | None = None,
    dataset: str | None = None,
    model: str | None = None,
) -> list[LeaderboardRow]:
    """Accuracy, latency and cost per approach, best accuracy first.

    Rows are compared against the `none` baseline *within* their own (dataset,
    model) cell, so a leaderboard spanning several datasets still reports each
    approach against the right baseline.
    """
    summaries = store.aggregate_by_approach(run_id=run_id, dataset=dataset, model=model)
    baselines = {
        (s.dataset, s.model): s for s in summaries if s.approach == BASELINE
    }

    rows = []
    for summary in summaries:
        baseline = baselines.get((summary.dataset, summary.model))
        rows.append(
            LeaderboardRow(
                approach=summary.approach,
                dataset=summary.dataset,
                model=summary.model,
                n=summary.n,
                correct=summary.correct,
                accuracy=summary.accuracy,
                delta_pp=(
                    (summary.accuracy - baseline.accuracy) * 100 if baseline else None
                ),
                errors=summary.errors,
                avg_latency_s=summary.avg_latency_s,
                avg_completion_tokens=summary.avg_completion_tokens,
                total_tokens=summary.total_tokens,
                cost_multiplier=(
                    summary.total_tokens / baseline.total_tokens
                    if baseline and baseline.total_tokens
                    else None
                ),
            )
        )
    return sorted(rows, key=lambda r: (r.dataset, r.model, -r.accuracy, r.approach))


_TABLE_COLUMNS = (
    ("approach", "<14"),
    ("acc", ">8"),
    ("delta", ">9"),
    ("errs", ">7"),
    ("lat(s)", ">10"),
    ("out tok", ">10"),
    ("cost x", ">9"),
)


def format_leaderboard(rows: list[LeaderboardRow]) -> str:
    """The console summary table, rendered from stored results."""
    header = "".join(f"{name:{spec}}" for name, spec in _TABLE_COLUMNS)
    lines = [header, "-" * len(header)]
    for row in rows:
        delta = "" if row.delta_pp is None else f"{row.delta_pp:+.1f}pp"
        cost = "" if row.cost_multiplier is None else f"{row.cost_multiplier:.1f}x"
        lines.append(
            f"{row.approach:<14}"
            f"{row.accuracy:>7.1%}"
            f"{delta:>9}"
            f"{row.errors:>7}"
            f"{row.avg_latency_s:>10.1f}"
            f"{row.avg_completion_tokens:>10.0f}"
            f"{cost:>9}"
        )
    return "\n".join(lines)


# -- per-query drill-down -------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    """What one approach did with one query."""

    approach: str
    predicted: str
    correct: bool
    error: str | None
    latency_s: float
    total_tokens: int
    router_approach: str | None
    response: str


@dataclass(frozen=True)
class QueryDetail:
    """One query, every approach's attempt at it, and who won."""

    query_id: str
    question: str
    gold: str
    attempts: list[Attempt]
    winner: str
    any_correct: bool


def query_detail(
    store: ResultsStore, *, run_id: str, dataset: str, model: str, query_id: str
) -> QueryDetail:
    """One query's prompt and gold answer, what every approach predicted, and
    which of them actually won."""
    results = store.results_for_query(
        run_id=run_id, dataset=dataset, model=model, query_id=query_id
    )
    label = label_query(results)
    return QueryDetail(
        query_id=query_id,
        question=results[0].question if results else "",
        gold=results[0].gold if results else "",
        attempts=[
            Attempt(
                approach=r.approach,
                predicted=r.predicted,
                correct=r.correct,
                error=r.error,
                latency_s=r.latency_s,
                total_tokens=r.total_tokens,
                router_approach=r.router_approach,
                response=r.response,
            )
            for r in results
        ],
        winner=label.winner,
        any_correct=label.any_correct,
    )


# -- a router vs. the actual winner ---------------------------------------


@dataclass(frozen=True)
class Confusion:
    """How often the router picked one approach when another actually won."""

    predicted: str
    actual: str
    count: int


@dataclass(frozen=True)
class Disagreement:
    """A single query where the router's pick was not the winner."""

    query_id: str
    dataset: str
    predicted: str
    actual: str
    any_correct: bool
    """False means no approach was correct, so `actual` is the baseline fallback."""


@dataclass(frozen=True)
class RouterComparison:
    """How one router did against the label rule.

    Agreement is judged only on queries some approach actually got right. Where
    nothing was correct, the label falls back to the baseline, so counting those
    as router mistakes would flatter or punish it for no reason; they are
    reported separately as `n_no_winner`.
    """

    router: str
    """Which router this is about - optillm's stock one, or the finetuned one."""
    n_queries: int
    n_judged: int
    n_no_winner: int
    agreements: int
    agreement_rate: float | None
    confusions: list[Confusion]
    disagreements: list[Disagreement]
    baseline_accuracy: float
    router_accuracy: float
    oracle_accuracy: float
    baseline_tokens: int
    router_tokens: int
    oracle_tokens: int


def _datasets_in(store: ResultsStore, run_id: str, model: str) -> list[str]:
    return sorted({r.dataset for r in store.query_results(run_id=run_id, model=model)})


def routers_compared(
    store: ResultsStore, *, run_id: str, model: str
) -> list[str]:
    """Which routers a run actually swept, stock first.

    The dashboard offers these as a choice rather than assuming both ran: a
    finetuned router only exists once there is a checkpoint to serve. Falls back
    to the stock router alone so a run with no router at all still renders the
    comparison's "nothing to judge" state instead of an empty picker.
    """
    swept = {
        summary.approach
        for summary in store.aggregate_by_approach(run_id=run_id, model=model)
    }
    return [router for router in ROUTERS if router in swept] or [ROUTER]


def router_comparison(
    store: ResultsStore,
    *,
    run_id: str,
    model: str,
    dataset: str | None = None,
    datasets: Sequence[str] | None = None,
    router: str = ROUTER,
) -> RouterComparison:
    """Compare one router's picks against the actual winners, with the headroom.

    Scope defaults to everything in the run. Narrow it with `dataset` for one,
    or `datasets` for an arbitrary group - which is how the per-category
    breakdown is built, since a category spans several datasets.

    `router` defaults to optillm's stock one, so every existing caller keeps
    comparing what it always compared. Pointing it at the finetuned router
    answers "did the finetune help?" from a single sweep, over identical queries,
    rather than by comparing two runs.

    `oracle_*` is what an always-right router would have scored and spent;
    `baseline_*` is what never routing at all would have. The router sits
    somewhere between, and the gap is the point of the whole exercise.
    """
    if dataset is not None:
        datasets = [dataset]
    elif datasets is None:
        datasets = _datasets_in(store, run_id, model)

    n_queries = n_judged = n_no_winner = agreements = 0
    confusions: Counter = Counter()
    disagreements: list[Disagreement] = []
    oracle_tokens = 0
    oracle_correct = 0

    for ds in datasets:
        costs = {
            (r.query_id, r.approach): r.total_tokens
            for r in store.query_results(run_id=run_id, dataset=ds, model=model)
        }
        for label in label_run(
            store, run_id=run_id, dataset=ds, model=model, router=router
        ):
            n_queries += 1
            oracle_tokens += costs.get((label.query_id, label.winner), 0)
            oracle_correct += label.any_correct

            if not label.any_correct:
                n_no_winner += 1
                continue
            if label.router_approach is None:
                continue

            n_judged += 1
            if label.router_approach == label.winner:
                agreements += 1
            else:
                confusions[(label.router_approach, label.winner)] += 1
                disagreements.append(
                    Disagreement(
                        query_id=label.query_id,
                        dataset=ds,
                        predicted=label.router_approach,
                        actual=label.winner,
                        any_correct=label.any_correct,
                    )
                )

    baseline = _approach_totals(store, run_id, model, datasets, BASELINE)
    routed = _approach_totals(store, run_id, model, datasets, router)

    return RouterComparison(
        router=router,
        n_queries=n_queries,
        n_judged=n_judged,
        n_no_winner=n_no_winner,
        agreements=agreements,
        agreement_rate=agreements / n_judged if n_judged else None,
        confusions=[
            Confusion(predicted=predicted, actual=actual, count=count)
            for (predicted, actual), count in confusions.most_common()
        ],
        disagreements=disagreements,
        baseline_accuracy=baseline.accuracy,
        router_accuracy=routed.accuracy,
        oracle_accuracy=oracle_correct / n_queries if n_queries else 0.0,
        baseline_tokens=baseline.total_tokens,
        router_tokens=routed.total_tokens,
        oracle_tokens=oracle_tokens,
    )


def comparisons_by_category(
    store: ResultsStore, *, run_id: str, model: str, router: str = ROUTER
) -> dict[str, RouterComparison]:
    """One comparison per task category, so it is visible whether the router is
    good everywhere or only on maths."""
    by_category: dict[str, list[str]] = {}
    for dataset in _datasets_in(store, run_id, model):
        # A dataset the harness no longer declares still has rows in old runs;
        # group it under its own name rather than dropping it from the report.
        category = category_of(dataset) if dataset in DATASETS else dataset
        by_category.setdefault(category, []).append(dataset)

    return {
        category: router_comparison(
            store, run_id=run_id, model=model, datasets=datasets, router=router
        )
        for category, datasets in sorted(by_category.items())
    }


@dataclass(frozen=True)
class _Totals:
    """One approach's accuracy and token spend over a group of datasets."""

    accuracy: float
    total_tokens: int


def _approach_totals(
    store: ResultsStore,
    run_id: str,
    model: str,
    datasets: Sequence[str],
    approach: str,
) -> _Totals:
    summaries = [
        s
        for ds in datasets
        for s in store.aggregate_by_approach(
            run_id=run_id, dataset=ds, model=model, approach=approach
        )
    ]
    n = sum(s.n for s in summaries)
    correct = sum(s.correct for s in summaries)
    return _Totals(
        accuracy=correct / n if n else 0.0,
        total_tokens=sum(s.total_tokens for s in summaries),
    )
