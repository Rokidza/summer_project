"""What a routing policy would actually have achieved - the metric that decides
whether a router is any good.

Agreement with the label is a poor score. A router that picks a *different*
approach which solved the same query at similar cost has done a perfect job and
scores zero for it, while one that predicts the baseline on a query only an
expensive approach solved scores zero too. The two are not the same event.

Because the store records, for every (query, approach), whether it was correct
and what it cost, the realised outcome of any policy is a table lookup: no GPU,
no serving stack, milliseconds. That is what this module does - and why the
dashboard is as legitimate a caller as the training loop, so it lives with the
results modules and never imports a deep-learning stack.

Two rules make the numbers comparable:

- **Scoring covers the queries with a complete approach matrix.** Under
  two-stage sweeping a query the baseline solved may carry the baseline row
  alone; a policy predicting an unrun approach there has a genuinely unknown
  outcome. It is excluded and reported as missing coverage, never imputed and
  never assumed wrong.
- **Cost is relative to the baseline over the same scored queries.** An
  absolute token total says nothing on its own; "3.4x the baseline" does.

Coverage is therefore not decoration: a policy scoring 0.9 over a fifth of the
queries has not been measured, and the score says so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

from router_lab.datasets import DATASETS
from router_lab.labels import (
    BASELINE,
    META_APPROACHES,
    ROUTER,
    is_win,
    winning_approach,
)
from router_lab.store import BenchmarkResult, ResultsStore

QueryKey = tuple[str, str]
"""(dataset, query_id) - how a query is identified across runs, exactly as the
training example builder identifies one. A bare query id is not unique: two
datasets both number their first problem 0."""

Policy = Mapping[QueryKey, str]
"""A routing decision per query. Missing keys are abstentions, not baselines."""

ALWAYS_BASELINE = "always_baseline"
BEST_SINGLE = "best_single"
ORACLE = "oracle"
STOCK_ROUTER = "stock_router"

REFERENCE_POLICIES = (ALWAYS_BASELINE, BEST_SINGLE, STOCK_ROUTER, ORACLE)
"""The lines every run is read against, weakest first.

A realised accuracy of 0.67 means nothing by itself; 0.67 sitting *below* an
always-baseline line at 0.72 is instantly legible as a failed run. Because the
label rule makes the baseline win every query it answers correctly, that line
sits high - which is the point.
"""

UNKNOWN_CATEGORY = "unknown"

MISROUTE_LIMIT = 10
MISROUTE_QUESTION_CHARS = 240


@dataclass(frozen=True)
class QueryOutcomes:
    """Every approach's recorded attempt at one query.

    Holds the result rows themselves rather than a digest of them, so the oracle
    can ask the label rule who won and get the same answer the training label
    was built from.
    """

    key: QueryKey
    question: str
    results: dict[str, BenchmarkResult]
    """By approach, including the meta-approaches - see `outcome`."""

    @property
    def dataset(self) -> str:
        return self.key[0]

    @property
    def category(self) -> str:
        """The dataset's grading category, or `unknown` for one since retired."""
        return DATASETS.get(self.dataset, {}).get("category", UNKNOWN_CATEGORY)

    @property
    def approaches(self) -> frozenset[str]:
        """The techniques run against this query - never the meta-approaches."""
        return frozenset(self.results) - META_APPROACHES

    @property
    def router_approach(self) -> str | None:
        """What optillm's pretrained router picked, if it ran on this query."""
        router = self.results.get(ROUTER)
        return router.router_approach if router else None

    def outcome(self, approach: str) -> BenchmarkResult | None:
        """The row a policy predicting `approach` would have realised.

        None means the outcome is unknown: the approach was not run here, or it
        is a meta-approach, whose own row is a routing decision rather than a
        technique's result. Scoring a router against another router's row would
        train the next router to defer.
        """
        if approach in META_APPROACHES:
            return None
        return self.results.get(approach)


@dataclass(frozen=True)
class OutcomeTable:
    """Recorded outcomes for a set of queries - the pure input to scoring."""

    queries: dict[QueryKey, QueryOutcomes] = field(default_factory=dict)

    @classmethod
    def from_results(cls, results: Iterable[BenchmarkResult]) -> "OutcomeTable":
        """Group rows by query, keeping the latest sweep's verdict for each.

        A query re-run in a later sweep, or run against a second served model,
        appears more than once. Pooling those would invent a query with two
        rows per approach, so the newest (run_id, model) cell wins - the same
        collapse the example builder makes, so labels and outcomes agree.
        """
        by_cell: dict[QueryKey, dict[tuple[str, str], dict[str, BenchmarkResult]]] = {}
        for result in results:
            key = (result.dataset, result.query_id)
            cell = (result.run_id, result.model)
            by_cell.setdefault(key, {}).setdefault(cell, {})[result.approach] = result

        queries = {}
        for key, cells in sorted(by_cell.items()):
            rows = cells[max(cells)]
            queries[key] = QueryOutcomes(
                key=key,
                question=next(iter(rows.values())).question,
                results=rows,
            )
        return cls(queries=queries)

    @property
    def approaches(self) -> frozenset[str]:
        """Every technique run anywhere in the table - the full matrix."""
        if not self.queries:
            return frozenset()
        return frozenset().union(
            *(query.approaches for query in self.queries.values())
        )

    def query_keys(self, *, dataset: str | None = None) -> list[QueryKey]:
        """The queries in the table, optionally narrowed to one dataset."""
        return [
            key for key in self.queries if dataset is None or key[0] == dataset
        ]

    def complete_keys(self, keys: Iterable[QueryKey] | None = None) -> list[QueryKey]:
        """The queries run through every technique the table knows about."""
        full = self.approaches
        return [
            key
            for key in (self.query_keys() if keys is None else keys)
            if key in self.queries and self.queries[key].approaches == full
        ]

    def select(self, keys: Iterable[QueryKey] | None) -> list[QueryKey]:
        """Requested keys that the table actually holds, in table order.

        `None` selects everything, which is what makes an unscoped score and a
        split-scoped one the same code path.
        """
        if keys is None:
            return self.query_keys()
        wanted = set(keys)
        return [key for key in self.queries if key in wanted]


@dataclass(frozen=True)
class PolicyScore:
    """What one policy realised, with the coverage that qualifies it."""

    name: str
    n_selected: int
    """Queries in scope, whether or not they could be scored."""
    n_scored: int
    n_correct: int
    n_errors: int
    """Scored queries whose chosen approach errored - counted as not correct."""
    total_tokens: int
    baseline_tokens: int
    """The baseline's cost over the same scored queries - the cost denominator."""
    category_accuracy: dict[str, float]
    n_incomplete: int = 0
    """Selected queries the sweep never ran through every approach."""
    n_abstained: int = 0
    """Selected queries the policy had no opinion about."""
    n_unrun: int = 0
    """Selected queries whose predicted approach was never run against them."""
    detail: str = ""
    """Which approach a derived policy turned out to be, e.g. best_single's."""

    @property
    def accuracy(self) -> float:
        """Realised accuracy: share of *scored* queries answered correctly."""
        return self.n_correct / self.n_scored if self.n_scored else 0.0

    @property
    def cost_multiplier(self) -> float:
        """Realised tokens as a multiple of always-baseline's, or 0 if unscored."""
        if not self.baseline_tokens:
            return 0.0
        return self.total_tokens / self.baseline_tokens

    @property
    def coverage(self) -> float:
        """Share of the selected queries whose outcome is actually known."""
        return self.n_scored / self.n_selected if self.n_selected else 0.0

    def why_uncovered(self) -> str:
        """Which of the three causes lost the coverage, counted.

        Coverage below 1 has three quite different meanings - the sweep did not
        finish the matrix, the policy had no opinion, or it predicted something
        never run - and a reader who cannot tell them apart cannot tell whether
        the number is the sweep's fault or the router's.
        """
        causes = {
            "incomplete matrix": self.n_incomplete,
            "no prediction": self.n_abstained,
            "prediction never run": self.n_unrun,
        }
        return ", ".join(f"{count} {cause}" for cause, count in causes.items() if count)

    def as_metrics(self) -> dict[str, float]:
        """Flat, primitive metrics a tracker can log as-is."""
        return {
            "realised_accuracy": self.accuracy,
            "cost_multiplier": self.cost_multiplier,
            "coverage": self.coverage,
            "realised_tokens": float(self.total_tokens),
        }

    def summary_line(self) -> str:
        detail = f" ({self.detail})" if self.detail else ""
        why = self.why_uncovered()
        return (
            f"{self.name}{detail}: accuracy {self.accuracy:.3f}  "
            f"cost {self.cost_multiplier:.2f}x  "
            f"coverage {self.coverage:.0%} ({self.n_scored}/{self.n_selected}"
            f"{'; ' + why if why else ''})"
        )


def score_policy(
    table: OutcomeTable,
    policy: Policy,
    *,
    name: str = "policy",
    keys: Iterable[QueryKey] | None = None,
    detail: str = "",
) -> PolicyScore:
    """Score `policy` over the selected queries, from recorded outcomes alone.

    A query is scored when it has a complete approach matrix, the policy has an
    opinion about it, and that opinion was actually run. Anything else is
    unknown and shows up as missing coverage.
    """
    selected = table.select(keys)
    scoreable = set(table.complete_keys(selected))

    n_correct = 0
    n_errors = 0
    total_tokens = 0
    baseline_tokens = 0
    per_category: dict[str, list[bool]] = {}
    n_scored = 0
    n_incomplete = 0
    n_abstained = 0
    n_unrun = 0

    for key in selected:
        if key not in scoreable:
            n_incomplete += 1
            continue
        query = table.queries[key]
        predicted = policy.get(key)
        if predicted is None:
            n_abstained += 1
            continue
        outcome = query.outcome(predicted)
        if outcome is None:
            n_unrun += 1
            continue
        baseline = query.outcome(BASELINE)

        n_scored += 1
        correct = is_win(outcome)
        n_correct += correct
        n_errors += outcome.error is not None
        total_tokens += outcome.total_tokens
        baseline_tokens += baseline.total_tokens if baseline else 0
        per_category.setdefault(query.category, []).append(correct)

    return PolicyScore(
        name=name,
        n_selected=len(selected),
        n_scored=n_scored,
        n_correct=n_correct,
        n_errors=n_errors,
        total_tokens=total_tokens,
        baseline_tokens=baseline_tokens,
        category_accuracy={
            category: sum(flags) / len(flags)
            for category, flags in per_category.items()
        },
        n_incomplete=n_incomplete,
        n_abstained=n_abstained,
        n_unrun=n_unrun,
        detail=detail,
    )


# -- the reference policies -----------------------------------------------


def always_policy(table: OutcomeTable, approach: str) -> dict[QueryKey, str]:
    """Predict one approach for everything."""
    return {key: approach for key in table.queries}


def oracle_policy(table: OutcomeTable) -> dict[QueryKey, str]:
    """The label rule's winner per query - the ceiling a router aims at.

    Its realised accuracy is the fraction of queries some approach solved, and
    because the rule takes the cheapest of those, it is also the cheapest policy
    that can reach that accuracy.
    """
    return {
        key: winning_approach(query.results.values())
        for key, query in table.queries.items()
    }


def stock_router_policy(table: OutcomeTable) -> dict[QueryKey, str]:
    """What optillm's pretrained router picked, where it ran.

    Queries it never saw are absent rather than defaulted: the comparison is
    against what the stock router *did*, and inventing a decision for it would
    flatter or damn it by accident.
    """
    return {
        key: query.router_approach
        for key, query in table.queries.items()
        if query.router_approach is not None
    }


def best_single_policy(
    table: OutcomeTable, *, keys: Iterable[QueryKey] | None = None
) -> tuple[str, dict[QueryKey, str]]:
    """The single approach that would have scored best on its own.

    Chosen by the same scoring function everything else goes through: most
    accurate, then cheapest, then alphabetical, so it never depends on the
    order rows came out of the database.
    """
    selected = list(table.select(keys))
    candidates = sorted(table.approaches)
    if not candidates:
        return BASELINE, {}
    scored = [
        (score_policy(table, always_policy(table, approach), keys=selected), approach)
        for approach in candidates
    ]
    # Sorted ascending on "worse", so `min` reads the tie-break in the order it
    # is stated: most accurate, then cheapest, then alphabetically first.
    best = min(
        scored, key=lambda pair: (-pair[0].accuracy, pair[0].total_tokens, pair[1])
    )[1]
    return best, always_policy(table, best)


def reference_scores(
    table: OutcomeTable, *, keys: Iterable[QueryKey] | None = None
) -> dict[str, PolicyScore]:
    """Score the four reference policies over the same queries as a run.

    Every one goes through `score_policy`, so a reference line and a model's
    point on the same chart were computed by the same code on the same queries.
    """
    selected = list(table.select(keys))
    best_approach, best_policy = best_single_policy(table, keys=selected)
    policies = {
        ALWAYS_BASELINE: (always_policy(table, BASELINE), ""),
        BEST_SINGLE: (best_policy, best_approach),
        STOCK_ROUTER: (stock_router_policy(table), ""),
        ORACLE: (oracle_policy(table), ""),
    }
    return {
        name: score_policy(
            table, policy, name=name, keys=selected, detail=detail
        )
        for name, (policy, detail) in policies.items()
    }


# -- diagnostics ----------------------------------------------------------


def macro_f1(labels: Sequence[str], predictions: Sequence[str]) -> float:
    """Unweighted mean F1 over the classes present, so rare ones still count.

    Pooled accuracy flatters a router that has learned to always predict the
    majority class, which on these labels is the baseline; this does not.
    """
    if not labels:
        return 0.0
    classes = sorted(set(labels) | set(predictions))
    scores = []
    for cls in classes:
        pairs = list(zip(labels, predictions))
        tp = sum(label == cls and pred == cls for label, pred in pairs)
        fp = sum(label != cls and pred == cls for label, pred in pairs)
        fn = sum(label == cls and pred != cls for label, pred in pairs)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def confusion_matrix(
    labels: Sequence[str], predictions: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Counts as `matrix[label][prediction]`, over the classes present."""
    classes = sorted(set(labels) | set(predictions))
    matrix = {label: {prediction: 0 for prediction in classes} for label in classes}
    for label, prediction in zip(labels, predictions):
        matrix[label][prediction] += 1
    return matrix


def format_confusion_matrix(matrix: Mapping[str, Mapping[str, int]]) -> str:
    """A fixed-width table: rows are labels, columns predictions.

    Text rather than a plot, deliberately - it stays readable in a terminal, in
    a log file, and in a tracker, and adds no plotting dependency to a run.
    """
    classes = sorted(matrix)
    if not classes:
        return "(nothing to compare)"
    width = max(max((len(c) for c in classes), default=0), 6)
    header = " " * (width + 2) + " ".join(c[:width].rjust(width) for c in classes)
    lines = ["rows: label   columns: prediction", header]
    for label in classes:
        cells = " ".join(
            str(matrix[label].get(prediction, 0)).rjust(width) for prediction in classes
        )
        lines.append(f"{label[:width].ljust(width)}  {cells}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Misroute:
    """One routing decision worth reading by eye, and what it actually cost."""

    key: QueryKey
    question: str
    predicted: str
    label: str
    predicted_correct: bool | None
    """None means the predicted approach was never run here - outcome unknown."""
    label_correct: bool
    predicted_tokens: int | None
    label_tokens: int
    kind: str

    @property
    def extra_tokens(self) -> int:
        """What the prediction cost over the label's approach; 0 if unknown."""
        if self.predicted_tokens is None:
            return 0
        return self.predicted_tokens - self.label_tokens


LOST_WIN = "lost win"
UNRUN_PREDICTION = "unrun prediction"
OVERSPEND = "overspend"

_SEVERITY = {LOST_WIN: 3, UNRUN_PREDICTION: 2, OVERSPEND: 1}


def worst_misroutes(
    table: OutcomeTable,
    policy: Policy,
    labels: Policy,
    *,
    keys: Iterable[QueryKey] | None = None,
    limit: int = MISROUTE_LIMIT,
) -> list[Misroute]:
    """The decisions that cost the most, worst first.

    Ranked by kind before magnitude: throwing away a win the label's approach
    achieved is worse than any amount of overspending on a query that was
    answered correctly anyway. This is the training-side equivalent of the
    dashboard's per-query drill-down - the thing that makes a bad number
    explicable instead of just bad.
    """
    misroutes = []
    for key in table.complete_keys(table.select(keys)):
        query = table.queries[key]
        predicted = policy.get(key)
        label = labels.get(key)
        if predicted is None or label is None or predicted == label:
            continue
        label_outcome = query.outcome(label)
        if label_outcome is None:
            continue
        predicted_outcome = query.outcome(predicted)
        predicted_correct = (
            None if predicted_outcome is None else is_win(predicted_outcome)
        )
        label_correct = is_win(label_outcome)
        misroute = Misroute(
            key=key,
            question=query.question,
            predicted=predicted,
            label=label,
            predicted_correct=predicted_correct,
            label_correct=label_correct,
            predicted_tokens=(
                None if predicted_outcome is None else predicted_outcome.total_tokens
            ),
            label_tokens=label_outcome.total_tokens,
            kind="",
        )
        kind = _kind(predicted_correct, label_correct, misroute.extra_tokens)
        if kind:
            misroutes.append(replace(misroute, kind=kind))
    misroutes.sort(
        key=lambda m: (-_SEVERITY[m.kind], -m.extra_tokens, m.key)
    )
    return misroutes[:limit]


def _kind(
    predicted_correct: bool | None, label_correct: bool, extra_tokens: int
) -> str:
    """Why a decision is worth looking at, or "" when it is not.

    Picking a *different* approach that solved the same query at no extra cost
    is not a mistake at all - it is the case this whole module exists for, and
    listing it as a misroute would reintroduce the label-agreement thinking the
    realised score replaces. Picking a losing approach on a query nothing solved
    is not the router's fault either.
    """
    if predicted_correct is None:
        return UNRUN_PREDICTION
    if label_correct and not predicted_correct:
        return LOST_WIN
    if label_correct and predicted_correct and extra_tokens > 0:
        return OVERSPEND
    return ""


def format_misroutes(misroutes: Sequence[Misroute]) -> str:
    """The misroutes as inspectable prose, for a tracker's text panel."""
    if not misroutes:
        return "No misroutes: every prediction matched the label or beat it."
    lines = []
    for misroute in misroutes:
        realised = (
            "not run"
            if misroute.predicted_correct is None
            else f"{'correct' if misroute.predicted_correct else 'wrong'}, "
            f"{misroute.predicted_tokens} tok"
        )
        lines.append(
            f"[{misroute.kind}] {misroute.key[0]}/{misroute.key[1]}\n"
            f"  predicted {misroute.predicted} ({realised})\n"
            f"  label     {misroute.label} "
            f"({'correct' if misroute.label_correct else 'wrong'}, "
            f"{misroute.label_tokens} tok)\n"
            f"  {_clip(misroute.question)}"
        )
    return "\n\n".join(lines)


def _clip(text: str, limit: int = MISROUTE_QUESTION_CHARS) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


# -- the store wrapper ----------------------------------------------------


def outcomes_for_runs(
    store: ResultsStore,
    *,
    run_ids: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    model: str | None = None,
) -> OutcomeTable:
    """Build an outcome table from a results database.

    The only store-touching function here, and the only one that is not pure -
    the same shape as the label rule, so scoring stays testable over results
    built by hand.

    Scope it the same way the training examples were scoped. The full approach
    matrix is derived from whatever is in the table, so pulling in a dataset the
    run never trained on can add an approach the rest of the queries were never
    run through, and make them look incomplete.
    """
    results = [
        result
        for run_id in (sorted(run_ids) if run_ids is not None else [None])
        for dataset in (sorted(datasets) if datasets is not None else [None])
        for result in store.query_results(run_id=run_id, dataset=dataset, model=model)
    ]
    return OutcomeTable.from_results(results)
