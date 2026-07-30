"""What one evaluation of a finetune records, with no model in sight.

The training loop hands over predictions it has already computed, so this layer
is testable without torch: examples in, tracker calls out. Whether the *numbers*
are right is `test_policy.py`'s job; what is asserted here is that a run ends up
with charts and text a failure can be read off.
"""

from router_lab.labels import BASELINE
from router_lab.policy import (
    ALWAYS_BASELINE,
    BEST_SINGLE,
    ORACLE,
    REFERENCE_POLICIES,
    STOCK_ROUTER,
)
from router_lab.training.evaluation import (
    MODEL_POLICY,
    as_policy,
    evaluate,
    labels_of,
    log_headline,
    references_for,
)
from router_lab.training.examples import TRAIN, VALIDATION, Example
from router_lab.training.optillm_router import APPROACHES, label_index
from tests.conftest import outcome_table as table_of, query_rows as rows
from tests.recording_tracker import RecordingTracker

import pytest


def example(query_id, approach, *, split=TRAIN, dataset="gsm8k") -> Example:
    return Example(
        query_id=query_id,
        dataset=dataset,
        text=f"question {query_id}",
        approach=approach,
        label=label_index(approach),
        split=split,
    )


def table():
    """Two queries the baseline solved, two only `bon` did."""
    return table_of(
        rows("1", none=(True, 100), bon=(True, 400), router=("bon", True, 420)),
        rows("2", none=(True, 100), bon=(True, 400), router=("none", True, 110)),
        rows("3", none=(False, 100), bon=(True, 400), router=("bon", True, 420)),
        rows("4", none=(False, 100), bon=(True, 400), router=("none", False, 110)),
    )


def examples():
    return [
        example("1", BASELINE),
        example("2", BASELINE),
        example("3", "bon"),
        example("4", "bon"),
    ]


PERFECT = [label_index(e.approach) for e in examples()]
ALL_BASELINE = [label_index(BASELINE)] * 4


def test_the_models_predictions_are_scored_as_a_routing_policy():
    tracker = RecordingTracker()

    evaluation = evaluate(
        examples(), PERFECT, table(), subset=TRAIN, epoch=1, tracker=tracker
    )

    assert evaluation.score.accuracy == 1.0
    assert tracker.metric_values(
        "realised_accuracy", subset=TRAIN, policy=MODEL_POLICY
    ) == [1.0]
    # Cheapest-correct on every query: 100 + 100 + 400 + 400 against a baseline
    # that costs 400, so exactly the oracle's cost.
    assert tracker.metric_values(
        "cost_multiplier", subset=TRAIN, policy=MODEL_POLICY
    ) == [2.5]
    assert tracker.metric_values("coverage", subset=TRAIN, policy=MODEL_POLICY) == [1.0]


def test_every_reference_policy_lands_on_the_models_own_chart():
    tracker = RecordingTracker()

    evaluate(examples(), ALL_BASELINE, table(), subset=TRAIN, epoch=1, tracker=tracker)

    policies = {
        context["policy"] for context in tracker.contexts_of("realised_accuracy")
    }
    assert policies == {MODEL_POLICY, *REFERENCE_POLICIES}
    assert tracker.metric_values(
        "realised_accuracy", subset=TRAIN, policy=ALWAYS_BASELINE
    ) == [0.5]
    assert tracker.metric_values(
        "realised_accuracy", subset=TRAIN, policy=ORACLE
    ) == [1.0]
    assert tracker.metric_values(
        "realised_accuracy", subset=TRAIN, policy=BEST_SINGLE
    ) == [1.0]
    assert tracker.metric_values(
        "realised_accuracy", subset=TRAIN, policy=STOCK_ROUTER
    ) == [0.75]
    # Predicting the baseline everywhere *is* the always-baseline policy, so the
    # model's line sits exactly on that reference - which is the comparison the
    # chart exists to make.
    assert tracker.metric_values(
        "realised_accuracy", subset=TRAIN, policy=MODEL_POLICY
    ) == [0.5]


def test_reference_lines_are_flat_across_the_run():
    tracker = RecordingTracker()
    outcomes = table()
    references = references_for(outcomes, examples())

    for epoch in (1, 2, 3):
        evaluate(
            examples(),
            ALL_BASELINE,
            outcomes,
            subset=VALIDATION,
            epoch=epoch,
            tracker=tracker,
            references=references,
        )

    assert tracker.metric_values(
        "realised_accuracy", subset=VALIDATION, policy=ORACLE
    ) == [1.0, 1.0, 1.0]
    assert [
        call.step
        for call in tracker.metrics
        if call.name == "realised_accuracy" and call.context["policy"] == ORACLE
    ] == [1, 2, 3]


def test_macro_f1_is_logged_beside_the_realised_score():
    tracker = RecordingTracker()

    evaluate(examples(), ALL_BASELINE, table(), subset=TRAIN, epoch=1, tracker=tracker)

    # Half right, but only ever one class predicted: macro-F1 sees that.
    assert tracker.metric_values("macro_f1", subset=TRAIN) == [
        pytest.approx((2 * 0.5 / 1.5) / 2)
    ]


def test_realised_accuracy_is_logged_per_task_category():
    tracker = RecordingTracker()
    outcomes = table_of(
        rows("1", dataset="gsm8k", none=(True, 100), bon=(True, 400)),
        rows("1", dataset="mbpp", none=(False, 100), bon=(True, 400)),
    )
    split = [example("1", BASELINE), example("1", "bon", dataset="mbpp")]

    evaluate(split, ALL_BASELINE[:2], outcomes, subset=TRAIN, epoch=1, tracker=tracker)

    assert tracker.metric_values(
        "category_accuracy", subset=TRAIN, category="numeric"
    ) == [1.0]
    assert tracker.metric_values(
        "category_accuracy", subset=TRAIN, category="code"
    ) == [0.0]


def test_a_confusion_matrix_is_logged_per_evaluation():
    tracker = RecordingTracker()

    evaluate(examples(), ALL_BASELINE, table(), subset=TRAIN, epoch=1, tracker=tracker)

    matrices = [call for call in tracker.texts if call.name == "confusion_matrix"]
    assert [call.step for call in matrices] == [1]
    assert matrices[0].context == {"subset": TRAIN}
    assert BASELINE in matrices[0].value and "bon" in matrices[0].value


def test_the_worst_misroutes_are_logged_as_inspectable_text():
    tracker = RecordingTracker()

    evaluation = evaluate(
        examples(), ALL_BASELINE, table(), subset=TRAIN, epoch=1, tracker=tracker
    )

    misroutes = [call for call in tracker.texts if call.name == "misroutes"]
    assert evaluation.n_misroutes == 2, "queries 3 and 4 were routed to the baseline"
    assert "question 3" in misroutes[0].value
    assert "lost win" in misroutes[0].value


def test_the_misroute_list_can_be_capped():
    tracker = RecordingTracker()

    evaluation = evaluate(
        examples(),
        ALL_BASELINE,
        table(),
        subset=TRAIN,
        epoch=1,
        tracker=tracker,
        misroute_limit=1,
    )

    assert evaluation.n_misroutes == 1


def test_predictions_that_do_not_line_up_are_refused():
    with pytest.raises(ValueError, match="line up"):
        as_policy(examples(), ALL_BASELINE[:2])


def test_predictions_become_approach_names_by_optillms_label_index():
    split = [example("1", BASELINE), example("2", BASELINE)]

    assert as_policy(split, [APPROACHES.index("moa"), APPROACHES.index("re2")]) == {
        ("gsm8k", "1"): "moa",
        ("gsm8k", "2"): "re2",
    }
    assert labels_of(split) == {("gsm8k", "1"): BASELINE, ("gsm8k", "2"): BASELINE}


def test_the_headline_summaries_state_the_score_and_what_to_read_it_against():
    tracker = RecordingTracker()
    outcomes = table()
    references = references_for(outcomes, examples())
    evaluation = evaluate(
        examples(), ALL_BASELINE, outcomes, subset=VALIDATION, epoch=2, tracker=tracker
    )

    log_headline(tracker, evaluation, references)

    assert tracker.summaries["final/subset"] == VALIDATION
    assert tracker.summaries["final/realised_accuracy"] == 0.5
    assert tracker.summaries["final/cost_multiplier"] == 1.0
    assert tracker.summaries["final/coverage"] == 1.0
    assert tracker.summaries[f"final/reference/{ORACLE}/realised_accuracy"] == 1.0
    assert tracker.summaries[f"final/reference/{BEST_SINGLE}/approach"] == "bon"
