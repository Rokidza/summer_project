"""The dashboard's data-access layer, against known SQLite fixtures.

These assert the shaped data the dashboard would render. The Streamlit
rendering layer itself is thin presentation and is not unit-tested.
"""

import pytest

from router_lab.labels import FINETUNED_ROUTER, ROUTER
from router_lab.views import (
    comparisons_by_category,
    format_leaderboard,
    leaderboard,
    query_detail,
    router_comparison,
    routers_compared,
)

from tests.conftest import build_result as make_result


def row(approach, query_id, *, correct, total_tokens, **overrides):
    return make_result(
        approach=approach,
        query_id=query_id,
        correct=correct,
        total_tokens=total_tokens,
        **overrides,
    )


# -- leaderboard ----------------------------------------------------------


@pytest.fixture
def two_approaches(store):
    store.write_results(
        [
            row("none", "1", correct=True, total_tokens=100, latency_s=1.0),
            row("none", "2", correct=False, total_tokens=100, latency_s=1.0),
            row("bon", "1", correct=True, total_tokens=400, latency_s=4.0),
            row("bon", "2", correct=True, total_tokens=400, latency_s=4.0),
        ]
    )
    return store


def test_leaderboard_reports_accuracy_per_approach(two_approaches):
    rows = {r.approach: r for r in leaderboard(two_approaches, run_id="run-1")}

    assert rows["none"].accuracy == 0.5
    assert rows["bon"].accuracy == 1.0


def test_leaderboard_ranks_the_best_approach_first(two_approaches):
    assert [r.approach for r in leaderboard(two_approaches, run_id="run-1")] == [
        "bon",
        "none",
    ]


def test_cost_is_a_multiplier_of_the_baselines_tokens(two_approaches):
    rows = {r.approach: r for r in leaderboard(two_approaches, run_id="run-1")}

    assert rows["none"].cost_multiplier == 1.0
    assert rows["bon"].cost_multiplier == 4.0  # 800 tokens vs the baseline's 200


def test_accuracy_delta_is_stated_against_the_baseline(two_approaches):
    rows = {r.approach: r for r in leaderboard(two_approaches, run_id="run-1")}

    assert rows["none"].delta_pp == 0.0
    assert rows["bon"].delta_pp == pytest.approx(50.0)


def test_latency_sits_alongside_accuracy(two_approaches):
    rows = {r.approach: r for r in leaderboard(two_approaches, run_id="run-1")}

    assert rows["bon"].avg_latency_s == 4.0


def test_cost_and_delta_are_absent_when_the_baseline_was_not_run(store):
    store.write_result(row("bon", "1", correct=True, total_tokens=400))

    only = leaderboard(store, run_id="run-1")[0]

    assert only.cost_multiplier is None
    assert only.delta_pp is None


def test_leaderboard_filters_by_dataset(store):
    store.write_results(
        [
            row("none", "1", correct=True, total_tokens=100, dataset="gsm8k"),
            row("none", "1", correct=False, total_tokens=100, dataset="boolq"),
        ]
    )

    rows = leaderboard(store, run_id="run-1", dataset="boolq")

    assert len(rows) == 1
    assert rows[0].accuracy == 0.0


def test_separate_runs_are_not_merged(store):
    store.write_results(
        [
            row("none", "1", correct=True, total_tokens=100, run_id="run-1"),
            row("none", "1", correct=False, total_tokens=100, run_id="run-2"),
        ]
    )

    assert leaderboard(store, run_id="run-1")[0].accuracy == 1.0
    assert leaderboard(store, run_id="run-2")[0].accuracy == 0.0


def test_the_console_table_shows_accuracy_delta_latency_and_cost(two_approaches):
    table = format_leaderboard(leaderboard(two_approaches, run_id="run-1"))

    header, _rule, best, *_ = table.splitlines()
    assert ["approach", "acc", "delta", "errs", "lat(s)", "out", "tok", "cost", "x"] == header.split()
    assert "bon" in best
    assert "100.0%" in best
    assert "+50.0pp" in best
    assert "4.0x" in best


# -- per-query drill-down -------------------------------------------------


def test_a_query_shows_its_prompt_gold_and_every_approachs_answer(two_approaches):
    detail = query_detail(
        two_approaches, run_id="run-1", dataset="gsm8k", model="qwen3-8b", query_id="2"
    )

    assert detail.question == "What is 2 + 2?"
    assert detail.gold == "4"
    assert {a.approach: a.correct for a in detail.attempts} == {
        "none": False,
        "bon": True,
    }
    assert detail.winner == "bon"


# -- router vs. the actual winner -----------------------------------------


@pytest.fixture
def router_run(store):
    """Query 1: router agrees. Query 2: router picks moa, bon actually won.
    Query 3: nothing was correct, so the label falls back to the baseline."""
    store.write_results(
        [
            row("none", "1", correct=False, total_tokens=100),
            row("bon", "1", correct=True, total_tokens=200),
            row("router", "1", correct=True, total_tokens=210, router_approach="bon"),
            row("none", "2", correct=False, total_tokens=100),
            row("bon", "2", correct=True, total_tokens=200),
            row("moa", "2", correct=False, total_tokens=900),
            row("router", "2", correct=False, total_tokens=910, router_approach="moa"),
            row("none", "3", correct=False, total_tokens=100),
            row("bon", "3", correct=False, total_tokens=200),
            row("router", "3", correct=False, total_tokens=210, router_approach="moa"),
        ]
    )
    return store


def test_agreement_rate_counts_only_genuine_router_mistakes(router_run):
    report = router_comparison(router_run, run_id="run-1", model="qwen3-8b")

    # Query 3 had no correct answer at all, so it is excluded from the rate.
    assert report.n_judged == 2
    assert report.agreements == 1
    assert report.agreement_rate == 0.5


def test_fallback_queries_are_reported_separately(router_run):
    report = router_comparison(router_run, run_id="run-1", model="qwen3-8b")

    assert report.n_no_winner == 1


def test_the_common_confusions_are_visible(router_run):
    report = router_comparison(router_run, run_id="run-1", model="qwen3-8b")

    assert report.confusions[0].predicted == "moa"
    assert report.confusions[0].actual == "bon"
    assert report.confusions[0].count == 1


def test_router_accuracy_and_cost_sit_between_the_baseline_and_the_oracle(router_run):
    report = router_comparison(router_run, run_id="run-1", model="qwen3-8b")

    assert report.baseline_accuracy == 0.0
    assert report.router_accuracy == pytest.approx(1 / 3)
    assert report.oracle_accuracy == pytest.approx(2 / 3)

    assert report.baseline_tokens == 300
    assert report.router_tokens == 1330
    # Oracle: bon on 1 and 2, baseline on 3 (nothing was correct).
    assert report.oracle_tokens == 500


def test_disagreeing_queries_can_be_drilled_into(router_run):
    report = router_comparison(router_run, run_id="run-1", model="qwen3-8b")

    assert [d.query_id for d in report.disagreements] == ["2"]
    assert report.disagreements[0].predicted == "moa"
    assert report.disagreements[0].actual == "bon"
    assert report.disagreements[0].any_correct is True


def test_comparison_can_be_narrowed_to_one_dataset(store):
    store.write_results(
        [
            row("none", "1", correct=False, total_tokens=100, dataset="gsm8k"),
            row("bon", "1", correct=True, total_tokens=200, dataset="gsm8k"),
            row("router", "1", correct=True, total_tokens=210,
                dataset="gsm8k", router_approach="bon"),
            row("none", "1", correct=False, total_tokens=100, dataset="boolq"),
            row("bon", "1", correct=True, total_tokens=200, dataset="boolq"),
            row("router", "1", correct=False, total_tokens=210,
                dataset="boolq", router_approach="moa"),
        ]
    )

    assert router_comparison(
        store, run_id="run-1", model="qwen3-8b", dataset="gsm8k"
    ).agreement_rate == 1.0
    assert router_comparison(
        store, run_id="run-1", model="qwen3-8b", dataset="boolq"
    ).agreement_rate == 0.0


def test_comparison_breaks_down_by_task_category(store):
    """gsm8k is numeric and boolq is boolean, so a router that is good only at
    maths has to show up as good only on one category."""
    store.write_results(
        [
            row("none", "1", correct=False, total_tokens=100, dataset="gsm8k"),
            row("bon", "1", correct=True, total_tokens=200, dataset="gsm8k"),
            row("router", "1", correct=True, total_tokens=210,
                dataset="gsm8k", router_approach="bon"),
            row("none", "1", correct=False, total_tokens=100, dataset="boolq"),
            row("bon", "1", correct=True, total_tokens=200, dataset="boolq"),
            row("router", "1", correct=False, total_tokens=210,
                dataset="boolq", router_approach="moa"),
        ]
    )

    by_category = comparisons_by_category(store, run_id="run-1", model="qwen3-8b")

    assert set(by_category) == {"numeric", "boolean"}
    assert by_category["numeric"].agreement_rate == 1.0
    assert by_category["boolean"].agreement_rate == 0.0


def test_several_datasets_of_one_category_are_pooled(store):
    store.write_results(
        [
            row("none", "1", correct=False, total_tokens=100, dataset="gsm8k"),
            row("bon", "1", correct=True, total_tokens=200, dataset="gsm8k"),
            row("router", "1", correct=True, total_tokens=210,
                dataset="gsm8k", router_approach="bon"),
            row("none", "1", correct=False, total_tokens=100, dataset="math500"),
            row("bon", "1", correct=True, total_tokens=200, dataset="math500"),
            row("router", "1", correct=False, total_tokens=210,
                dataset="math500", router_approach="moa"),
        ]
    )

    by_category = comparisons_by_category(store, run_id="run-1", model="qwen3-8b")

    assert set(by_category) == {"numeric"}
    assert by_category["numeric"].n_judged == 2
    assert by_category["numeric"].agreement_rate == 0.5


def test_a_dataset_the_harness_no_longer_declares_still_reports(store):
    """Old runs reference datasets that may since have been renamed or dropped."""
    store.write_results(
        [
            row("none", "1", correct=False, total_tokens=100, dataset="retired_bench"),
            row("bon", "1", correct=True, total_tokens=200, dataset="retired_bench"),
            row("router", "1", correct=True, total_tokens=210,
                dataset="retired_bench", router_approach="bon"),
        ]
    )

    by_category = comparisons_by_category(store, run_id="run-1", model="qwen3-8b")

    assert by_category["retired_bench"].agreement_rate == 1.0


def test_comparison_over_a_run_with_no_router_results_is_empty_not_an_error(store):
    store.write_result(row("none", "1", correct=True, total_tokens=100))

    report = router_comparison(store, run_id="run-1", model="qwen3-8b")

    assert report.n_judged == 0
    assert report.agreement_rate is None


# -- stock router vs. the finetuned one -----------------------------------
#
# Both are swept as ordinary approaches over identical queries, so "did the
# finetune help?" is answerable from one run rather than by comparing two.


@pytest.fixture
def both_routers(store):
    """Two queries; `bon` wins query 1, the baseline wins query 2.

    The stock router picks moa both times (wrong twice); the finetuned one picks
    what actually won.
    """
    store.write_results(
        [
            row("none", "1", correct=False, total_tokens=100),
            row("bon", "1", correct=True, total_tokens=200),
            row("moa", "1", correct=False, total_tokens=900),
            row("router", "1", correct=False, total_tokens=910, router_approach="moa"),
            row("router_ft", "1", correct=True, total_tokens=220,
                router_approach="bon"),
            row("none", "2", correct=True, total_tokens=100),
            row("bon", "2", correct=True, total_tokens=200),
            row("moa", "2", correct=True, total_tokens=900),
            row("router", "2", correct=True, total_tokens=910, router_approach="moa"),
            row("router_ft", "2", correct=True, total_tokens=110,
                router_approach="none"),
        ]
    )
    return store


def test_the_comparison_defaults_to_the_stock_router(both_routers):
    report = router_comparison(both_routers, run_id="run-1", model="qwen3-8b")

    assert report.router == ROUTER
    assert report.agreement_rate == 0.0


def test_the_same_run_can_be_read_as_the_finetuned_routers_comparison(both_routers):
    report = router_comparison(
        both_routers, run_id="run-1", model="qwen3-8b", router=FINETUNED_ROUTER
    )

    assert report.router == FINETUNED_ROUTER
    assert report.agreement_rate == 1.0
    assert report.disagreements == []
    # The same queries and the same winners - only the predictions differ.
    assert report.n_queries == report.n_judged == 2
    assert report.oracle_tokens == 300


def test_each_routers_own_accuracy_and_spend_are_reported(both_routers):
    stock = router_comparison(both_routers, run_id="run-1", model="qwen3-8b")
    finetuned = router_comparison(
        both_routers, run_id="run-1", model="qwen3-8b", router=FINETUNED_ROUTER
    )

    assert (stock.router_accuracy, stock.router_tokens) == (0.5, 1820)
    assert (finetuned.router_accuracy, finetuned.router_tokens) == (1.0, 330)
    assert stock.baseline_tokens == finetuned.baseline_tokens == 200


def test_the_per_category_breakdown_follows_the_chosen_router(both_routers):
    by_category = comparisons_by_category(
        both_routers, run_id="run-1", model="qwen3-8b", router=FINETUNED_ROUTER
    )

    assert [report.router for report in by_category.values()] == [FINETUNED_ROUTER]
    assert by_category["numeric"].agreement_rate == 1.0


def test_the_routers_a_run_swept_are_listed_stock_first(both_routers):
    assert routers_compared(both_routers, run_id="run-1", model="qwen3-8b") == [
        ROUTER,
        FINETUNED_ROUTER,
    ]


def test_only_the_routers_that_actually_ran_are_offered(store):
    store.write_results(
        [
            row("none", "1", correct=True, total_tokens=100),
            row("router_ft", "1", correct=True, total_tokens=110,
                router_approach="none"),
        ]
    )

    assert routers_compared(store, run_id="run-1", model="qwen3-8b") == [
        FINETUNED_ROUTER
    ]


def test_a_run_with_no_router_at_all_still_offers_the_stock_one(store):
    """So the comparison renders its "nothing to judge" state, not an empty picker."""
    store.write_result(row("none", "1", correct=True, total_tokens=100))

    assert routers_compared(store, run_id="run-1", model="qwen3-8b") == [ROUTER]
