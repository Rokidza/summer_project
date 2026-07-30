"""Scoring a finetune on realised outcomes, and recording why the number is what
it is.

The scoring itself is `router_lab.policy` - deliberately outside this package,
because a policy's realised outcome is a fact about results rather than about
training, and the dashboard wants it too. What lives here is the translation:
the model's predictions over a split become a policy, the policy gets scored,
and the score plus its diagnostics go to the tracker.

Nothing here imports torch. The training loop hands over predictions it has
already computed, so this module - and its tests - need no model.

What a run gets per evaluation:

- `realised_accuracy` and `cost_multiplier`, under a `policy` context, sharing a
  chart with the four reference policies. The references are constant over the
  run and re-logged at every step, so they draw flat lines: a model curve
  crossing above `always_baseline` is the whole point of the chart.
- `macro_f1`, because pooled agreement flatters a majority-class predictor.
- `category_accuracy` per grading category, so "good at maths, useless at code"
  is visible rather than averaged away.
- a confusion matrix and the worst misroutes as text, so a bad number can be
  read rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from router_lab.policy import (
    OutcomeTable,
    PolicyScore,
    QueryKey,
    confusion_matrix,
    format_confusion_matrix,
    format_misroutes,
    macro_f1,
    reference_scores,
    score_policy,
    worst_misroutes,
)
from router_lab.training.examples import Example
from router_lab.training.optillm_router import APPROACHES
from router_lab.training.tracker import Tracker

MODEL_POLICY = "model"
"""The context value distinguishing the run's own curve from the references."""


@dataclass(frozen=True)
class Evaluation:
    """One split scored at one point in a run."""

    subset: str
    epoch: int
    score: PolicyScore
    macro_f1: float
    n_misroutes: int

    def report(self) -> str:
        return (
            f"{self.subset:<10} realised {self.score.accuracy:.3f}  "
            f"cost {self.score.cost_multiplier:.2f}x  "
            f"macro-F1 {self.macro_f1:.3f}  "
            f"coverage {self.score.coverage:.0%}"
        )


def keys_of(examples: Sequence[Example]) -> list[QueryKey]:
    """The queries a split covers, in the outcome table's terms."""
    return [(example.dataset, example.query_id) for example in examples]


def as_policy(
    examples: Sequence[Example], predictions: Sequence[int]
) -> dict[QueryKey, str]:
    """Head outputs over a split, as a routing decision per query."""
    if len(examples) != len(predictions):
        raise ValueError(
            f"{len(predictions)} predictions for {len(examples)} examples: "
            f"they have to line up, or the policy is scored against the wrong queries"
        )
    return {
        key: APPROACHES[prediction]
        for key, prediction in zip(keys_of(examples), predictions)
    }


def labels_of(examples: Sequence[Example]) -> dict[QueryKey, str]:
    return {key: e.approach for key, e in zip(keys_of(examples), examples)}


def references_for(
    outcomes: OutcomeTable, examples: Sequence[Example]
) -> dict[str, PolicyScore]:
    """The reference lines for one split - constant across a run, so computed once."""
    return reference_scores(outcomes, keys=keys_of(examples))


def evaluate(
    examples: Sequence[Example],
    predictions: Sequence[int],
    outcomes: OutcomeTable,
    *,
    subset: str,
    epoch: int,
    tracker: Tracker,
    references: Mapping[str, PolicyScore] | None = None,
    misroute_limit: int | None = None,
) -> Evaluation:
    """Score the model's predictions over one split and log the lot."""
    policy = as_policy(examples, predictions)
    labels = labels_of(examples)
    keys = keys_of(examples)
    score = score_policy(outcomes, policy, name=MODEL_POLICY, keys=keys)

    context = {"subset": subset}
    _log_score(tracker, score, subset=subset, epoch=epoch)
    if references is None:
        references = references_for(outcomes, examples)
    for reference in references.values():
        _log_score(tracker, reference, subset=subset, epoch=epoch)

    label_names = [labels[key] for key in keys]
    predicted_names = [policy[key] for key in keys]
    f1 = macro_f1(label_names, predicted_names)
    tracker.log_metric("macro_f1", f1, step=epoch, context=context)
    for category, accuracy in sorted(score.category_accuracy.items()):
        tracker.log_metric(
            "category_accuracy",
            accuracy,
            step=epoch,
            context={**context, "category": category},
        )

    tracker.log_text(
        "confusion_matrix",
        format_confusion_matrix(confusion_matrix(label_names, predicted_names)),
        step=epoch,
        context=context,
    )
    misroutes = worst_misroutes(
        outcomes,
        policy,
        labels,
        keys=keys,
        **({} if misroute_limit is None else {"limit": misroute_limit}),
    )
    tracker.log_text(
        "misroutes", format_misroutes(misroutes), step=epoch, context=context
    )
    return Evaluation(
        subset=subset,
        epoch=epoch,
        score=score,
        macro_f1=f1,
        n_misroutes=len(misroutes),
    )


def _log_score(
    tracker: Tracker, score: PolicyScore, *, subset: str, epoch: int
) -> None:
    """One policy's metrics at one step, tagged with whose they are."""
    context = {"subset": subset, "policy": score.name}
    for name, value in score.as_metrics().items():
        tracker.log_metric(name, value, step=epoch, context=context)


def log_headline(
    tracker: Tracker,
    evaluation: Evaluation,
    references: Mapping[str, PolicyScore] | None = None,
    *,
    prefix: str = "final",
) -> None:
    """Record the numbers a run is judged by as summaries, not just curves.

    A run's row in the tracker's table has to say whether the router was any
    good; realised accuracy and cost multiplier are that, and the references go
    beside them because neither means anything alone.
    """
    tracker.log_summary(f"{prefix}/subset", evaluation.subset)
    tracker.log_summary(f"{prefix}/realised_accuracy", evaluation.score.accuracy)
    tracker.log_summary(f"{prefix}/cost_multiplier", evaluation.score.cost_multiplier)
    tracker.log_summary(f"{prefix}/coverage", evaluation.score.coverage)
    tracker.log_summary(f"{prefix}/macro_f1", evaluation.macro_f1)
    tracker.log_summary(f"{prefix}/n_scored", evaluation.score.n_scored)
    for name, reference in (references or {}).items():
        tracker.log_summary(f"{prefix}/reference/{name}/realised_accuracy", reference.accuracy)
        tracker.log_summary(f"{prefix}/reference/{name}/cost_multiplier", reference.cost_multiplier)
        if reference.detail:
            tracker.log_summary(f"{prefix}/reference/{name}/approach", reference.detail)
