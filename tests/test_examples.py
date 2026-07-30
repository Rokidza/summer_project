"""Building training examples out of a real results database."""

import pytest

from router_lab.datasets import Problem
from router_lab.grading import system_prompt
from router_lab.harness import SweepSettings, run_sweep
from router_lab.labels import LABEL_RULE_VERSION
from router_lab.training.examples import (
    SPLITS,
    TRAIN,
    build_examples,
)
from router_lab.training.optillm_router import (
    APPROACHES,
    INFERENCE_EFFORT,
    build_input_text,
)

from tests.fake_server import FakeInferenceServer, completion


def seed_cell(store, make_result, *, n_queries=20, run_id="run-1", dataset="gsm8k"):
    """One (run, dataset, model) cell where `bon` wins the odd queries.

    The baseline answers the even ones correctly and is cheaper, so the label
    rule hands it those; on the odd ones only `bon` is correct.
    """
    for i in range(n_queries):
        baseline_correct = i % 2 == 0
        store.write_result(
            make_result(
                run_id=run_id,
                dataset=dataset,
                approach="none",
                query_id=str(i),
                question=f"Question {i}?",
                correct=baseline_correct,
                total_tokens=100,
            )
        )
        store.write_result(
            make_result(
                run_id=run_id,
                dataset=dataset,
                approach="bon",
                query_id=str(i),
                question=f"Question {i}?",
                correct=True,
                total_tokens=500,
            )
        )


def splits_by_query(built) -> dict[tuple[str, str], str]:
    """Split per query, keyed the way the splitter keys them: dataset and id."""
    return {(e.dataset, e.query_id): e.split for e in built.examples}


def test_input_text_is_the_serving_plugins_construction(store, make_result):
    seed_cell(store, make_result, n_queries=2)

    examples = build_examples(store).examples

    expected = f"{system_prompt('numeric')}\n\nUser: Question 0?"
    assert [e.text for e in examples if e.query_id == "0"] == [expected]


def test_input_text_is_built_from_what_the_harness_actually_sent(store):
    """The other half of the drift guard.

    `tests/test_optillm_router.py` pins the serving plugin's construction; this
    pins the *harness* side, by running a real sweep against the fake server and
    rebuilding the text from the messages that went over the wire. Change how
    the harness composes a request and this fails, rather than training quietly
    drifting away from the text the sweep was graded on.
    """
    problem = Problem(id="1", question="What is 2 + 2?", gold="4")
    with FakeInferenceServer(default_reply=completion("\\boxed{4}")) as server:
        run_sweep(
            store,
            run_id="run-1",
            dataset="gsm8k",
            approaches=["none"],
            problems=[problem],
            settings=SweepSettings(
                model="qwen3-8b", base_url=server.base_url, retries=0, concurrency=1
            ),
        )
        sent = server.requests[0]["messages"]

    on_the_wire = {message["role"]: message["content"] for message in sent}
    built = build_examples(store).examples[0]
    assert built.text == build_input_text(on_the_wire["system"], on_the_wire["user"])


def test_effort_is_pinned_to_the_inference_constant(store, make_result):
    seed_cell(store, make_result, n_queries=4)

    assert {e.effort for e in build_examples(store).examples} == {INFERENCE_EFFORT}


def test_labels_are_indices_into_optillms_approach_order(store, make_result):
    seed_cell(store, make_result, n_queries=4)

    by_query = {e.query_id: e for e in build_examples(store).examples}

    assert by_query["0"].approach == "none"
    assert by_query["0"].label == APPROACHES.index("none") == 0
    assert by_query["1"].approach == "bon"
    assert by_query["1"].label == APPROACHES.index("bon")


def test_an_approach_outside_optillms_label_space_is_dropped(store, make_result):
    seed_cell(store, make_result, n_queries=2)
    # Cheaper than the baseline and correct, so the label rule would pick it -
    # but the pretrained head has no output for it.
    store.write_result(
        make_result(
            run_id="run-1",
            approach="made_up",
            query_id="1",
            correct=True,
            total_tokens=1,
        )
    )

    built = build_examples(store)

    assert [e.query_id for e in built.examples] == ["0"]
    assert built.provenance.n_dropped == 1


def test_the_same_seed_gives_the_same_split(store, make_result):
    seed_cell(store, make_result)

    first = {e.query_id: e.split for e in build_examples(store, seed=7).examples}
    second = {e.query_id: e.split for e in build_examples(store, seed=7).examples}
    other = {e.query_id: e.split for e in build_examples(store, seed=8).examples}

    assert first == second
    assert first != other


def test_splits_partition_the_queries(store, make_result):
    seed_cell(store, make_result, n_queries=200)

    built = build_examples(store, seed=0)

    per_split = {name: {e.query_id for e in built.split(name)} for name in SPLITS}
    assert set().union(*per_split.values()) == {str(i) for i in range(200)}
    assert sum(len(ids) for ids in per_split.values()) == 200
    # Shares are approximate by construction (see `assign_split`), so this pins
    # the ballpark, not an exact slice.
    assert 0.6 < len(per_split[TRAIN]) / 200 < 0.8
    assert built.provenance.split_counts == {
        name: len(ids) for name, ids in per_split.items()
    }


def test_growing_the_database_never_moves_a_querys_split(store, make_result):
    """Otherwise a query held out of one finetune trains the next one."""
    seed_cell(store, make_result, n_queries=20)
    before = splits_by_query(build_examples(store, seed=0))

    seed_cell(store, make_result, n_queries=40, run_id="run-2", dataset="mmlu")
    after = splits_by_query(build_examples(store, seed=0))

    assert all(after[key] == split for key, split in before.items())
    assert len(after) == 60


def test_the_same_query_keeps_one_split_across_runs(store, make_result):
    """Splits are per query, so re-running a dataset cannot leak across them."""
    seed_cell(store, make_result, n_queries=20, run_id="run-1")
    seed_cell(store, make_result, n_queries=20, run_id="run-2")

    built = build_examples(store, seed=0)

    assert len(built.examples) == 20
    assert built.provenance.run_ids == ["run-1", "run-2"]
    assert built.provenance.n_queries == 20, "coverage must count queries, not cells"
    assert built.provenance.coverage_fraction == 1.0


def test_provenance_travels_with_the_built_set(store, make_result):
    seed_cell(store, make_result, n_queries=10, dataset="gsm8k")
    seed_cell(store, make_result, n_queries=4, dataset="mmlu")

    provenance = build_examples(store, seed=3).provenance

    assert provenance.run_ids == ["run-1"]
    assert provenance.models == ["qwen3-8b"]
    assert provenance.dataset_counts == {"gsm8k": 10, "mmlu": 4}
    assert provenance.n_examples == 14
    assert provenance.coverage_fraction == 1.0
    assert provenance.split_seed == 3
    assert provenance.label_rule_version == LABEL_RULE_VERSION
    assert provenance.fingerprint


def test_partial_coverage_is_reported_never_imputed(store, make_result):
    """A two-stage sweep leaves baseline-solved queries with the baseline alone."""
    seed_cell(store, make_result, n_queries=4)
    store.write_result(
        make_result(run_id="run-1", approach="none", query_id="99", correct=True)
    )

    built = build_examples(store)

    assert built.provenance.coverage_fraction == pytest.approx(4 / 5)
    assert {e.query_id for e in built.examples} == {"0", "1", "2", "3", "99"}


def test_queries_no_approach_solved_are_labelled_but_counted(store, make_result):
    """The rule's baseline fallback is a label, not evidence the baseline worked."""
    seed_cell(store, make_result, n_queries=2)
    for approach in ("none", "bon"):
        store.write_result(
            make_result(
                run_id="run-1", approach=approach, query_id="42", correct=False
            )
        )

    built = build_examples(store)

    unsolved = next(e for e in built.examples if e.query_id == "42")
    assert unsolved.approach == "none"
    assert built.provenance.n_unsolved == 1


def test_provenance_records_the_models_actually_read(store, make_result):
    """Not the filter that selected them - two models collapse to one example."""
    seed_cell(store, make_result, n_queries=4)
    store.write_result(
        make_result(run_id="run-1", model="gemma-3-1b", approach="none", query_id="0")
    )

    built = build_examples(store)

    assert built.provenance.models == ["gemma-3-1b", "qwen3-8b"]
    assert len(built.examples) == 4


def test_the_fingerprint_changes_with_the_content(store, make_result):
    seed_cell(store, make_result, n_queries=4)
    before = build_examples(store).provenance.fingerprint

    store.write_result(
        make_result(
            run_id="run-1", approach="bon", query_id="0", correct=True, total_tokens=1
        )
    )

    assert build_examples(store).provenance.fingerprint != before


def test_a_run_can_be_selected_by_id(store, make_result):
    seed_cell(store, make_result, n_queries=4, run_id="run-1")
    seed_cell(store, make_result, n_queries=4, run_id="run-2", dataset="mmlu")

    built = build_examples(store, run_ids=["run-2"])

    assert built.provenance.run_ids == ["run-2"]
    assert built.provenance.dataset_counts == {"mmlu": 4}


def test_provenance_is_loggable_as_flat_parameters(store, make_result):
    seed_cell(store, make_result, n_queries=4)

    params = build_examples(store).provenance.as_params()

    assert params["examples/run_ids"] == "run-1"
    assert params["examples/dataset_counts"] == "gsm8k=4"
    assert params["examples/label_rule_version"] == LABEL_RULE_VERSION
