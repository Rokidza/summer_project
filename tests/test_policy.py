"""What a routing policy would actually have achieved, over constructed results.

The scoring path is pure - a table of recorded outcomes in, a score out - so it
is tested the way the label rule is: results built by hand, no I/O, no model.
Only the store wrapper at the bottom touches a database, and that one gets a
real temp-file SQLite rather than a fake.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from router_lab.labels import BASELINE, ROUTER
from router_lab.policy import (
    ALWAYS_BASELINE,
    BEST_SINGLE,
    ORACLE,
    STOCK_ROUTER,
    always_policy,
    best_single_policy,
    confusion_matrix,
    format_confusion_matrix,
    format_misroutes,
    macro_f1,
    oracle_policy,
    outcomes_for_runs,
    reference_scores,
    score_policy,
    stock_router_policy,
    worst_misroutes,
)
from tests.conftest import outcome_table as table_of, query_rows as rows


# -- the score itself ------------------------------------------------------


def test_scores_realised_accuracy_and_token_cost():
    table = table_of(
        rows("1", none=(False, 100), bon=(True, 400)),
        rows("2", none=(True, 120), bon=(True, 500)),
    )
    score = score_policy(table, {("gsm8k", "1"): "bon", ("gsm8k", "2"): BASELINE})

    assert score.accuracy == 1.0
    assert score.total_tokens == 400 + 120
    assert score.coverage == 1.0
    assert score.n_scored == 2


def test_cost_multiplier_compares_against_the_baseline_on_the_same_queries():
    table = table_of(
        rows("1", none=(False, 100), bon=(True, 400)),
        rows("2", none=(True, 100), bon=(True, 400)),
    )
    score = score_policy(table, always_policy(table, "bon"))

    assert score.baseline_tokens == 200
    assert score.total_tokens == 800
    assert score.cost_multiplier == 4.0


def test_always_baseline_cost_multiplier_is_exactly_one():
    table = table_of(
        rows("1", none=(False, 100), bon=(True, 400)),
        rows("2", none=(True, 123), bon=(False, 700)),
    )
    references = reference_scores(table)

    assert references[ALWAYS_BASELINE].cost_multiplier == 1.0
    assert references[ALWAYS_BASELINE].accuracy == 0.5


def test_an_errored_result_counts_as_run_but_never_as_correct():
    table = table_of(
        rows("1", none=(False, 100), bon=(True, 400, "context length exceeded")),
    )
    score = score_policy(table, always_policy(table, "bon"))

    assert score.n_scored == 1
    assert score.accuracy == 0.0
    assert score.n_errors == 1
    assert score.total_tokens == 400


def test_a_prediction_never_run_is_excluded_and_lowers_coverage():
    table = table_of(
        rows("1", none=(False, 100), bon=(True, 400)),
        rows("2", none=(True, 100), bon=(True, 400)),
    )
    # mcts appears nowhere in the results: the outcome is unknown, not wrong.
    score = score_policy(table, {("gsm8k", "1"): "mcts", ("gsm8k", "2"): "bon"})

    assert score.n_selected == 2
    assert score.n_scored == 1
    assert score.coverage == 0.5
    assert score.accuracy == 1.0


def test_a_query_missing_from_the_policy_is_excluded_and_lowers_coverage():
    table = table_of(
        rows("1", none=(True, 100), bon=(True, 400)),
        rows("2", none=(True, 100), bon=(True, 400)),
    )
    score = score_policy(table, {("gsm8k", "1"): "bon"})

    assert score.n_scored == 1
    assert score.coverage == 0.5


def test_a_partial_approach_matrix_is_excluded_from_the_score():
    # What a two-stage sweep leaves behind: the baseline solved query 2, so
    # stage two never ran the expensive approaches against it.
    table = table_of(
        rows("1", none=(False, 100), bon=(True, 400)),
        rows("2", none=(True, 100)),
    )
    score = score_policy(table, always_policy(table, BASELINE))

    assert score.n_selected == 2
    assert score.n_scored == 1
    assert score.coverage == 0.5
    # Query 2's baseline row is not scored even though it *was* run: scoring the
    # queries with a complete matrix is what makes policies comparable.
    assert score.total_tokens == 100
    assert score.accuracy == 0.0


def test_a_query_no_approach_solved_scores_zero_for_every_policy():
    table = table_of(
        rows("1", none=(False, 100), bon=(False, 400)),
        rows("2", none=(True, 100), bon=(True, 400)),
    )
    references = reference_scores(table)

    assert references[ORACLE].accuracy == 0.5
    assert references[ALWAYS_BASELINE].accuracy == 0.5
    assert references[BEST_SINGLE].accuracy == 0.5


# -- the reference policies -----------------------------------------------


def test_oracle_accuracy_is_the_fraction_with_any_correct_approach():
    table = table_of(
        rows("1", none=(False, 100), bon=(True, 400)),
        rows("2", none=(False, 100), bon=(False, 400)),
        rows("3", none=(True, 100), bon=(True, 400)),
        rows("4", none=(False, 100), bon=(True, 400)),
    )
    references = reference_scores(table)

    assert references[ORACLE].accuracy == 0.75
    # The cheapest correct approach on each query, which is the label rule's
    # winner - so the oracle is also the cheapest policy that can score 0.75.
    assert references[ORACLE].total_tokens == 400 + 100 + 100 + 400


def test_oracle_falls_back_to_the_baseline_when_nothing_worked():
    table = table_of(rows("1", none=(False, 100), bon=(False, 400)))

    assert oracle_policy(table) == {("gsm8k", "1"): BASELINE}


def test_oracle_breaks_cost_ties_the_way_the_label_rule_does():
    table = table_of(rows("1", bon=(True, 400), moa=(True, 400)))

    # Equal tokens, equal calls, equal latency: the approach name decides, so
    # the oracle and the training label can never disagree.
    assert oracle_policy(table) == {("gsm8k", "1"): "bon"}


def test_best_single_picks_the_most_accurate_approach_and_names_it():
    table = table_of(
        rows("1", none=(True, 100), bon=(True, 400), moa=(False, 900)),
        rows("2", none=(False, 100), bon=(True, 400), moa=(False, 900)),
    )
    score = reference_scores(table)[BEST_SINGLE]

    assert score.detail == "bon"
    assert score.accuracy == 1.0
    assert score.cost_multiplier == 4.0


def test_best_single_prefers_the_cheaper_approach_when_accuracy_ties():
    table = table_of(
        rows("1", none=(True, 100), bon=(True, 400)),
        rows("2", none=(True, 100), bon=(True, 400)),
    )
    name, policy = best_single_policy(table)

    assert name == BASELINE
    assert set(policy.values()) == {BASELINE}


def test_stock_router_is_scored_on_the_outcome_of_what_it_picked():
    table = table_of(
        rows("1", none=(False, 100), bon=(True, 400), router=("bon", True, 420)),
        rows("2", none=(True, 100), bon=(False, 400), router=("bon", False, 410)),
    )
    assert stock_router_policy(table) == {
        ("gsm8k", "1"): "bon",
        ("gsm8k", "2"): "bon",
    }
    score = reference_scores(table)[STOCK_ROUTER]

    # Scored on bon's own rows, not on the router's - so the number is
    # comparable with every other policy's, on identical accounting.
    assert score.accuracy == 0.5
    assert score.total_tokens == 800
    assert score.cost_multiplier == 4.0


def test_a_query_the_stock_router_never_ran_lowers_its_coverage_only():
    table = table_of(
        rows("1", none=(False, 100), bon=(True, 400), router=("bon", True, 420)),
        rows("2", none=(True, 100), bon=(True, 400)),
    )
    references = reference_scores(table)

    assert references[STOCK_ROUTER].coverage == 0.5
    assert references[ORACLE].coverage == 1.0


def test_the_routers_own_row_is_never_a_scoreable_outcome():
    table = table_of(
        rows("1", none=(False, 100), bon=(True, 400), router=("bon", True, 420)),
    )
    # A policy predicting "router" is asking a router to defer to a router; it
    # is unknown, not scored.
    score = score_policy(table, {("gsm8k", "1"): ROUTER})

    assert score.n_scored == 0
    assert score.coverage == 0.0
    # ... and the router's presence does not make the matrix incomplete.
    assert reference_scores(table)[ORACLE].coverage == 1.0


# -- scoping ---------------------------------------------------------------


def test_scoring_can_be_scoped_to_a_subset_of_queries():
    table = table_of(
        rows("1", none=(True, 100), bon=(True, 400)),
        rows("2", none=(False, 100), bon=(True, 400)),
    )
    score = score_policy(
        table, always_policy(table, BASELINE), keys=[("gsm8k", "2")]
    )

    assert score.n_selected == 1
    assert score.accuracy == 0.0


def test_scoring_can_be_scoped_to_one_dataset():
    table = table_of(
        rows("1", dataset="gsm8k", none=(True, 100), bon=(True, 400)),
        rows("1", dataset="mmlu", none=(False, 100), bon=(True, 400)),
    )
    score = score_policy(
        table, always_policy(table, BASELINE), keys=table.keys(dataset="mmlu")
    )

    assert score.n_selected == 1
    assert score.accuracy == 0.0


def test_keys_outside_the_table_are_not_selected():
    table = table_of(rows("1", none=(True, 100), bon=(True, 400)))
    score = score_policy(
        table, always_policy(table, BASELINE), keys=[("gsm8k", "1"), ("gsm8k", "99")]
    )

    assert score.n_selected == 1
    assert score.coverage == 1.0


def test_realised_accuracy_is_broken_down_by_task_category():
    table = table_of(
        rows("1", dataset="gsm8k", none=(True, 100), bon=(True, 400)),
        rows("2", dataset="gsm8k", none=(True, 100), bon=(False, 400)),
        rows("1", dataset="mbpp", none=(False, 100), bon=(False, 400)),
    )
    score = score_policy(table, always_policy(table, "bon"))

    assert score.category_accuracy == {"numeric": 0.5, "code": 0.0}


def test_an_empty_selection_scores_zero_rather_than_dividing_by_it():
    table = table_of(rows("1", none=(True, 100), bon=(True, 400)))
    score = score_policy(table, {}, keys=[])

    assert (score.n_selected, score.n_scored) == (0, 0)
    assert score.accuracy == 0.0
    assert score.coverage == 0.0
    assert score.cost_multiplier == 0.0


def test_score_flattens_to_metrics_a_tracker_can_log():
    table = table_of(rows("1", none=(False, 100), bon=(True, 400)))
    metrics = score_policy(table, always_policy(table, "bon")).as_metrics()

    assert metrics["realised_accuracy"] == 1.0
    assert metrics["cost_multiplier"] == 4.0
    assert metrics["coverage"] == 1.0
    assert metrics["realised_tokens"] == 400


# -- diagnostics ----------------------------------------------------------


def test_macro_f1_averages_over_classes_not_over_examples():
    # Nine of ten are the majority class and predicted perfectly; the minority
    # class is missed entirely. Pooled accuracy says 0.9, macro-F1 does not.
    labels = ["none"] * 9 + ["bon"]
    predictions = ["none"] * 10

    assert macro_f1(labels, predictions) == pytest.approx(
        (2 * 0.9 / 1.9 + 0.0) / 2
    )


def test_macro_f1_of_a_perfect_predictor_is_one():
    assert macro_f1(["none", "bon"], ["none", "bon"]) == 1.0


def test_macro_f1_of_nothing_is_zero():
    assert macro_f1([], []) == 0.0


def test_confusion_matrix_counts_label_against_prediction():
    matrix = confusion_matrix(
        ["none", "none", "bon"], ["none", "bon", "bon"]
    )

    assert matrix == {
        "none": {"none": 1, "bon": 1},
        "bon": {"none": 0, "bon": 1},
    }
    text = format_confusion_matrix(matrix)
    assert "none" in text and "bon" in text


def test_worst_misroutes_puts_lost_wins_before_overspending():
    table = table_of(
        # Routed to an approach that failed where the label's approach worked.
        rows("1", none=(False, 100), bon=(True, 400)),
        # Routed correctly, but paid four times the label's cost to do it.
        rows("2", none=(True, 100), bon=(True, 400)),
        # Routed exactly right.
        rows("3", none=(True, 100), bon=(False, 400)),
    )
    policy = {("gsm8k", "1"): BASELINE, ("gsm8k", "2"): "bon", ("gsm8k", "3"): BASELINE}
    labels = {("gsm8k", "1"): "bon", ("gsm8k", "2"): BASELINE, ("gsm8k", "3"): BASELINE}

    misroutes = worst_misroutes(table, policy, labels)

    assert [m.key for m in misroutes] == [("gsm8k", "1"), ("gsm8k", "2")]
    assert misroutes[0].predicted == BASELINE
    assert misroutes[0].label == "bon"
    assert misroutes[0].predicted_correct is False
    assert misroutes[1].extra_tokens == 300

    text = format_misroutes(misroutes)
    assert "question 1" in text
    assert "question 3" not in text


def test_worst_misroutes_reports_an_unrun_prediction_as_unknown():
    table = table_of(rows("1", none=(True, 100), bon=(True, 400)))
    misroutes = worst_misroutes(
        table, {("gsm8k", "1"): "mcts"}, {("gsm8k", "1"): BASELINE}
    )

    assert misroutes[0].predicted_correct is None
    assert "not run" in format_misroutes(misroutes)


def test_worst_misroutes_is_capped():
    table = table_of(
        *[rows(str(i), none=(False, 100), bon=(True, 400)) for i in range(10)]
    )
    policy = {("gsm8k", str(i)): BASELINE for i in range(10)}
    labels = {("gsm8k", str(i)): "bon" for i in range(10)}

    assert len(worst_misroutes(table, policy, labels, limit=3)) == 3


def test_format_misroutes_says_so_when_there_are_none():
    assert "no misroutes" in format_misroutes([]).lower()


# -- the store wrapper ----------------------------------------------------


def test_outcomes_for_runs_reads_a_real_database(store):
    store.write_results(
        rows("1", none=(False, 100), bon=(True, 400))
        + rows("2", none=(True, 120), bon=(True, 500))
    )
    table = outcomes_for_runs(store)

    assert reference_scores(table)[ORACLE].accuracy == 1.0
    assert table.approaches == frozenset({BASELINE, "bon"})


def test_outcomes_for_runs_keeps_the_latest_runs_verdict(store):
    store.write_results(
        rows("1", run_id="run-1", none=(False, 100), bon=(False, 400))
        + rows("1", run_id="run-2", none=(True, 100), bon=(False, 400))
    )
    table = outcomes_for_runs(store)

    # Same query, two sweeps: the later one is what happened, and the two are
    # never pooled into one query with four rows.
    assert reference_scores(table)[ALWAYS_BASELINE].accuracy == 1.0
    assert reference_scores(table)[ALWAYS_BASELINE].n_scored == 1


def test_outcomes_for_runs_filters_by_run_dataset_and_model(store):
    store.write_results(
        rows("1", run_id="run-1", dataset="gsm8k", none=(True, 100), bon=(True, 400))
        + rows("1", run_id="run-2", dataset="mmlu", none=(True, 100), bon=(True, 400))
        + rows(
            "1", run_id="run-3", model="other", none=(True, 100), bon=(True, 400)
        )
    )
    assert outcomes_for_runs(store, run_ids=["run-1"]).keys() == [("gsm8k", "1")]
    assert outcomes_for_runs(store, datasets=["mmlu"]).keys() == [("mmlu", "1")]
    assert outcomes_for_runs(store, model="other").keys() == [("gsm8k", "1")]
    assert len(outcomes_for_runs(store, model="qwen3-8b").keys()) == 2


def test_outcomes_for_runs_on_an_empty_database_scores_nothing(store):
    table = outcomes_for_runs(store)

    assert table.keys() == []
    assert reference_scores(table)[ORACLE].n_selected == 0


def test_scoring_needs_no_deep_learning_stack():
    """The dashboard and a cluster-side sweep import this; neither has torch.

    Asserted in a subprocess because the training tests have already imported
    torch into this one.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, router_lab.policy; "
            "sys.exit(1 if 'torch' in sys.modules else 0)",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
