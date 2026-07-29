"""Results-store seam: exercised against a real temp-file SQLite database.

Per the platform spec, SQLite is fast enough to use for real, so these tests
never mock the database - they assert on what was actually stored.
"""

import threading

import pytest

from router_lab.store import ResultsStore, SweepConfig

from tests.conftest import build_result as make_result


def test_written_result_reads_back_with_every_field_intact(db_path):
    written = make_result(error="Timeout", router_approach="moa")

    with ResultsStore.open(db_path) as store:
        store.write_result(written)

    with ResultsStore.open(db_path) as store:
        assert store.query_results() == [written]


def test_reopening_an_existing_database_preserves_earlier_results(db_path):
    with ResultsStore.open(db_path) as store:
        store.write_result(make_result(query_id="1"))

    with ResultsStore.open(db_path) as store:
        store.write_result(make_result(query_id="2"))

    with ResultsStore.open(db_path) as store:
        assert {r.query_id for r in store.query_results()} == {"1", "2"}


def test_rewriting_the_same_cell_replaces_it_rather_than_duplicating(db_path):
    with ResultsStore.open(db_path) as store:
        store.write_result(make_result(correct=False, predicted="5"))
        store.write_result(make_result(correct=True, predicted="4"))

        results = store.query_results()

    assert len(results) == 1
    assert results[0].predicted == "4"


@pytest.fixture
def populated(db_path):
    """Two runs x two datasets x two approaches x two models, one query each."""
    with ResultsStore.open(db_path) as store:
        for run_id in ("run-1", "run-2"):
            for dataset in ("gsm8k", "boolq"):
                for approach in ("none", "bon"):
                    for model in ("qwen3-8b", "gemma-3-1b"):
                        store.write_result(
                            make_result(
                                run_id=run_id,
                                dataset=dataset,
                                approach=approach,
                                model=model,
                                query_id="1",
                            )
                        )
        yield store


def test_results_can_be_filtered_by_a_single_facet(populated):
    results = populated.query_results(dataset="boolq")

    assert len(results) == 8
    assert {r.dataset for r in results} == {"boolq"}


def test_filters_combine_conjunctively(populated):
    results = populated.query_results(
        run_id="run-2", dataset="gsm8k", approach="bon", model="qwen3-8b"
    )

    assert len(results) == 1
    assert (results[0].run_id, results[0].dataset) == ("run-2", "gsm8k")


def test_all_approaches_for_one_query_are_fetched_together(populated):
    results = populated.results_for_query(
        run_id="run-1", dataset="gsm8k", model="qwen3-8b", query_id="1"
    )

    assert {r.approach for r in results} == {"none", "bon"}


def test_aggregates_report_accuracy_latency_and_tokens_per_cell(db_path):
    with ResultsStore.open(db_path) as store:
        store.write_results(
            [
                make_result(
                    query_id="1",
                    correct=True,
                    latency_s=1.0,
                    completion_tokens=10,
                    total_tokens=30,
                    call_count=1,
                ),
                make_result(
                    query_id="2",
                    correct=False,
                    latency_s=3.0,
                    completion_tokens=20,
                    total_tokens=70,
                    call_count=4,
                    error="Timeout",
                ),
            ]
        )

        summaries = store.aggregate_by_approach(run_id="run-1")

    assert len(summaries) == 1
    summary = summaries[0]
    assert (summary.approach, summary.dataset, summary.model) == (
        "bon",
        "gsm8k",
        "qwen3-8b",
    )
    assert (summary.n, summary.correct, summary.accuracy) == (2, 1, 0.5)
    assert summary.errors == 1
    assert summary.avg_latency_s == 2.0
    assert summary.avg_completion_tokens == 15.0
    assert summary.total_tokens == 100
    assert summary.total_calls == 5


def test_aggregates_split_by_approach_dataset_and_model(populated):
    summaries = populated.aggregate_by_approach(run_id="run-1")

    cells = {(s.approach, s.dataset, s.model) for s in summaries}
    assert len(cells) == 8
    assert all(s.n == 1 for s in summaries)


def test_runs_are_listed_with_what_they_covered(populated):
    runs = populated.list_runs()

    assert [r.run_id for r in runs] == ["run-1", "run-2"]
    assert runs[0].datasets == ["boolq", "gsm8k"]
    assert runs[0].models == ["gemma-3-1b", "qwen3-8b"]
    assert runs[0].approaches == ["bon", "none"]
    assert runs[0].n_results == 8


def test_query_ids_are_distinct_and_numerically_ordered(db_path):
    with ResultsStore.open(db_path) as store:
        for query_id in ("10", "2", "2"):
            for approach in ("none", "bon"):
                store.write_result(make_result(query_id=query_id, approach=approach))

        ids = store.query_ids(run_id="run-1", dataset="gsm8k", model="qwen3-8b")

    assert ids == ["2", "10"]


def build_config(**overrides) -> SweepConfig:
    fields = dict(
        run_id="run-1",
        dataset="gsm8k",
        model="qwen3-8b",
        mode="two-stage",
        pool_size=200,
        approaches=["none", "bon", "moa"],
        control_fraction=0.1,
        seed=7,
    )
    fields.update(overrides)
    return SweepConfig(**fields)


def test_sweep_configuration_reads_back_for_a_run_id(db_path):
    written = build_config()

    with ResultsStore.open(db_path) as store:
        store.record_sweep_config(written)

    with ResultsStore.open(db_path) as store:
        assert store.sweep_configs(run_id="run-1") == [written]


def test_a_single_stage_sweep_records_no_control_fraction_or_seed(store):
    store.record_sweep_config(
        build_config(mode="single", control_fraction=None, seed=None)
    )

    config = store.sweep_configs(run_id="run-1")[0]
    assert config.mode == "single"
    assert config.control_fraction is None
    assert config.seed is None


def test_one_run_holds_a_configuration_per_dataset(store):
    store.record_sweep_config(build_config(dataset="gsm8k", pool_size=200))
    store.record_sweep_config(build_config(dataset="boolq", pool_size=50))

    configs = store.sweep_configs(run_id="run-1")
    assert [(c.dataset, c.pool_size) for c in configs] == [
        ("boolq", 50),
        ("gsm8k", 200),
    ]


def test_re_recording_the_same_cell_replaces_its_configuration(store):
    store.record_sweep_config(build_config(pool_size=200))
    store.record_sweep_config(build_config(pool_size=500))

    assert [c.pool_size for c in store.sweep_configs(run_id="run-1")] == [500]


def test_configurations_of_other_runs_are_not_returned(store):
    store.record_sweep_config(build_config(run_id="run-1"))
    store.record_sweep_config(build_config(run_id="run-2"))

    assert [c.run_id for c in store.sweep_configs(run_id="run-2")] == ["run-2"]


def test_approach_coverage_is_derived_from_the_stored_results(store):
    """Coverage is never written down - it is counted off the rows themselves,
    so it cannot drift away from them."""
    store.write_results(
        [
            make_result(query_id="1", approach="none"),
            make_result(query_id="1", approach="bon"),
            make_result(query_id="2", approach="none"),
        ]
    )

    coverage = store.approach_coverage(
        run_id="run-1", dataset="gsm8k", model="qwen3-8b"
    )
    assert coverage == {"1": ["bon", "none"], "2": ["none"]}


def test_approach_coverage_of_an_empty_cell_is_empty(store):
    assert store.approach_coverage(run_id="nope", dataset="gsm8k", model="x") == {}


def test_a_store_can_be_used_from_another_thread(db_path):
    """Streamlit runs each rerun on a fresh thread, and the harness writes from
    the thread that drains its worker pool - so a store outlives its creator."""
    with ResultsStore.open(db_path) as store:
        store.write_result(make_result(query_id="1"))

        failures = []

        def use_from_another_thread():
            try:
                store.write_result(make_result(query_id="2"))
                assert len(store.query_results()) == 2
                assert store.aggregate_by_approach()[0].n == 2
                assert store.list_runs()[0].n_results == 2
            except Exception as exc:  # noqa: BLE001 - reported, not raised, across the boundary
                failures.append(f"{type(exc).__name__}: {exc}")

        thread = threading.Thread(target=use_from_another_thread)
        thread.start()
        thread.join()

    assert failures == []


def test_concurrent_writers_do_not_lose_results(db_path):
    with ResultsStore.open(db_path) as store:
        def write(start):
            store.write_results(
                [make_result(query_id=str(start + i)) for i in range(20)]
            )

        threads = [threading.Thread(target=write, args=(n * 20,)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(store.query_results()) == 80
