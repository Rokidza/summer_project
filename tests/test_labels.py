"""The router label rule - a pure function over one query's results.

No database, no network: every case is a hand-constructed set of results.
"""

from router_lab.labels import (
    BASELINE,
    FINETUNED_ROUTER,
    ROUTER,
    label_query,
    label_run,
    winning_approach,
)
from router_lab.store import ResultsStore

from tests.conftest import build_result as make_result


def result(approach, *, correct, total_tokens=100, **overrides):
    return make_result(
        approach=approach, correct=correct, total_tokens=total_tokens, **overrides
    )


def test_cheapest_correct_approach_wins():
    results = [
        result("none", correct=False, total_tokens=50),
        result("moa", correct=True, total_tokens=900),
        result("bon", correct=True, total_tokens=300),
    ]

    assert winning_approach(results) == "bon"


def test_baseline_wins_when_no_approach_is_correct():
    results = [
        result("none", correct=False, total_tokens=50),
        result("bon", correct=False, total_tokens=300),
    ]

    assert winning_approach(results) == BASELINE


def test_baseline_alone_wins():
    assert winning_approach([result("none", correct=True, total_tokens=50)]) == BASELINE


def test_correct_baseline_beats_a_pricier_correct_approach():
    results = [
        result("none", correct=True, total_tokens=50),
        result("bon", correct=True, total_tokens=300),
    ]

    assert winning_approach(results) == BASELINE


def test_ties_on_cost_resolve_deterministically():
    tied = [
        result("moa", correct=True, total_tokens=300),
        result("bon", correct=True, total_tokens=300),
    ]

    assert winning_approach(tied) == winning_approach(list(reversed(tied)))


def test_a_cheaper_errored_result_is_never_the_winner():
    results = [
        result("none", correct=False, total_tokens=50),
        result("rto", correct=False, total_tokens=0, error="Timeout"),
        result("bon", correct=True, total_tokens=300),
    ]

    assert winning_approach(results) == "bon"


def test_an_errored_result_marked_correct_is_still_not_selected():
    results = [
        result("rto", correct=True, total_tokens=1, error="Timeout"),
        result("bon", correct=True, total_tokens=300),
    ]

    assert winning_approach(results) == "bon"


def test_call_count_breaks_a_token_tie():
    results = [
        result("moa", correct=True, total_tokens=300, call_count=5),
        result("bon", correct=True, total_tokens=300, call_count=2),
    ]

    assert winning_approach(results) == "bon"


def test_label_records_the_fallback_case_distinguishably():
    correct = label_query([result("bon", correct=True), result("none", correct=False)])
    fallback = label_query(
        [result("bon", correct=False), result("none", correct=False)]
    )

    assert (correct.winner, correct.any_correct) == ("bon", True)
    assert (fallback.winner, fallback.any_correct) == (BASELINE, False)


def test_the_router_is_never_the_winner_it_is_the_thing_being_judged():
    """`router` is a meta-approach: it picks a technique rather than being one.

    Letting it win its own comparison would make the label circular and give a
    future trained router "use the other router" as a target.
    """
    results = [
        result("none", correct=False, total_tokens=50),
        result("router", correct=True, total_tokens=90, router_approach="bon"),
        result("bon", correct=True, total_tokens=300),
    ]

    assert winning_approach(results) == "bon"


def test_the_baseline_still_wins_when_only_the_router_answered_correctly():
    results = [
        result("none", correct=False, total_tokens=50),
        result("router", correct=True, total_tokens=90, router_approach="moa"),
    ]

    label = label_query(results)
    assert label.winner == BASELINE
    assert label.any_correct is False


def test_label_carries_what_the_router_predicted():
    label = label_query(
        [
            result("none", correct=True, total_tokens=50),
            result("router", correct=True, total_tokens=90, router_approach="moa"),
        ]
    )

    assert label.router_approach == "moa"


def test_the_finetuned_router_is_never_the_winner_either():
    """It is trained on these labels, so letting it win would close the loop."""
    results = [
        result("none", correct=False, total_tokens=50),
        result(FINETUNED_ROUTER, correct=True, total_tokens=90, router_approach="bon"),
        result("bon", correct=True, total_tokens=300),
    ]

    assert winning_approach(results) == "bon"
    assert label_query(results).any_correct is True


def test_the_baseline_still_wins_when_only_the_finetuned_router_answered():
    results = [
        result("none", correct=False, total_tokens=50),
        result(FINETUNED_ROUTER, correct=True, total_tokens=90, router_approach="moa"),
    ]

    label = label_query(results)
    assert label.winner == BASELINE
    assert label.any_correct is False


def test_which_routers_prediction_is_reported_is_the_callers_choice():
    """One sweep carrying both routers is two comparisons over identical queries."""
    results = [
        result("none", correct=True, total_tokens=50),
        result(ROUTER, correct=True, total_tokens=90, router_approach="moa"),
        result(FINETUNED_ROUTER, correct=True, total_tokens=60, router_approach="none"),
    ]

    stock = label_query(results)
    finetuned = label_query(results, router=FINETUNED_ROUTER)

    assert (stock.router, stock.router_approach) == (ROUTER, "moa")
    assert (finetuned.router, finetuned.router_approach) == (FINETUNED_ROUTER, "none")
    assert stock.winner == finetuned.winner == BASELINE


def test_a_router_that_did_not_run_predicted_nothing_rather_than_something():
    results = [
        result("none", correct=True, total_tokens=50),
        result(ROUTER, correct=True, total_tokens=90, router_approach="moa"),
    ]

    assert label_query(results, router=FINETUNED_ROUTER).router_approach is None


def test_labels_are_produced_for_every_query_in_a_run(tmp_path):
    with ResultsStore.open(tmp_path / "results.sqlite") as store:
        store.write_results(
            [
                result("none", correct=False, query_id="1"),
                result("bon", correct=True, query_id="1", total_tokens=300),
                result("none", correct=True, query_id="2", total_tokens=50),
                result("bon", correct=True, query_id="2", total_tokens=300),
            ]
        )

        labels = label_run(store, run_id="run-1", dataset="gsm8k", model="qwen3-8b")

    assert {label.query_id: label.winner for label in labels} == {
        "1": "bon",
        "2": BASELINE,
    }
