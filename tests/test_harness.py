"""Eval-harness seam: the HTTP boundary between the harness and optillm.

Every test here points the harness at a fake OpenAI-compatible server (see
`tests.fake_server`) and asserts on what landed in a real SQLite store.
"""

import sqlite3

import pytest

from router_lab.datasets import Problem
from router_lab.harness import SweepSettings, run_sweep, run_two_stage_sweep
from router_lab.store import ResultsStore

from tests.fake_server import FakeInferenceServer, completion

PROBLEM = Problem(id="1", question="What is 2 + 2?", gold="4")

POOL = [Problem(id=str(n), question=f"query {n}?", gold="4") for n in range(1, 7)]


@pytest.fixture
def store(tmp_path):
    with ResultsStore.open(tmp_path / "results.sqlite") as store:
        yield store


def settings_for(server, **overrides):
    fields = dict(
        model="qwen3-8b",
        base_url=server.base_url,
        api_key="sk-test",
        concurrency=2,
        max_tokens=64,
        temperature=0.0,
        timeout=10.0,
        retries=0,
    )
    fields.update(overrides)
    return SweepSettings(**fields)


def sweep(
    store,
    server,
    *,
    approaches=("none",),
    problems=(PROBLEM,),
    resume=True,
    **overrides,
):
    run_sweep(
        store,
        run_id="run-1",
        dataset="gsm8k",
        approaches=list(approaches),
        problems=list(problems),
        settings=settings_for(server, **overrides),
        show_progress=False,
        resume=resume,
    )


def test_one_row_is_written_per_query_and_approach(store):
    problems = [PROBLEM, Problem(id="2", question="3 + 3?", gold="6")]

    with FakeInferenceServer() as server:
        sweep(store, server, approaches=("none", "bon"), problems=problems)

    written = store.query_results()
    assert len(written) == 4
    assert {(r.query_id, r.approach) for r in written} == {
        ("1", "none"),
        ("1", "bon"),
        ("2", "none"),
        ("2", "bon"),
    }


def test_every_stored_field_is_populated_from_the_response(store):
    with FakeInferenceServer() as server:
        server.default_reply = completion(
            "so it is \\boxed{4}", completion_tokens=12, total_tokens=30, llm_calls=3
        )
        sweep(store, server)

    result = store.query_results()[0]
    assert result.run_id == "run-1"
    assert result.dataset == "gsm8k"
    assert result.model == "qwen3-8b"
    assert result.approach == "none"
    assert result.query_id == "1"
    assert result.question == "What is 2 + 2?"
    assert result.correct is True
    assert result.predicted == "4"
    assert result.gold == "4"
    assert result.latency_s > 0
    assert result.completion_tokens == 12
    assert result.total_tokens == 30
    assert result.call_count == 3
    assert result.error is None
    assert result.response == "so it is \\boxed{4}"


def test_a_wrong_answer_is_recorded_as_incorrect(store):
    with FakeInferenceServer() as server:
        server.default_reply = completion("\\boxed{5}")
        sweep(store, server)

    assert store.query_results()[0].correct is False


def test_the_routers_predicted_approach_is_recorded(store):
    with FakeInferenceServer() as server:
        server.default_reply = completion(router_approach="moa")
        sweep(store, server, approaches=("router",))

    assert store.query_results()[0].router_approach == "moa"


def test_the_baseline_asks_for_no_optillm_approach(store):
    with FakeInferenceServer() as server:
        sweep(store, server, approaches=("none",))
        requests = server.requests

    assert "optillm_approach" not in requests[0]


def test_a_named_approach_is_requested_by_name(store):
    with FakeInferenceServer() as server:
        sweep(store, server, approaches=("bon",))
        requests = server.requests

    assert requests[0]["optillm_approach"] == "bon"


def test_total_tokens_falls_back_to_completion_tokens_when_absent(store):
    """optillm's own usage block reports completion tokens only."""
    with FakeInferenceServer() as server:
        server.default_reply = completion(completion_tokens=40, total_tokens=None)
        sweep(store, server)

    assert store.query_results()[0].total_tokens == 40


def test_call_count_defaults_to_one_when_the_server_does_not_report_it(store):
    with FakeInferenceServer() as server:
        server.default_reply = completion(llm_calls=None)
        sweep(store, server)

    assert store.query_results()[0].call_count == 1


def test_a_transient_failure_is_retried_and_the_retry_is_recorded(store):
    with FakeInferenceServer() as server:
        server.replies = [500, completion("\\boxed{4}")]
        sweep(store, server, retries=1, retry_backoff_s=0.0)

    result = store.query_results()[0]
    assert result.error is None
    assert result.correct is True


def test_exhausted_retries_leave_a_row_carrying_the_error(store):
    with FakeInferenceServer() as server:
        server.replies = [500, 500]
        sweep(store, server, retries=1, retry_backoff_s=0.0)

    result = store.query_results()[0]
    assert result.correct is False
    assert result.error is not None
    assert result.predicted == ""
    assert result.gold == "4"


def test_two_sweeps_into_one_database_stay_distinguishable_by_run(store, tmp_path):
    with FakeInferenceServer() as server:
        sweep(store, server)
        run_sweep(
            store,
            run_id="run-2",
            dataset="gsm8k",
            approaches=["none"],
            problems=[PROBLEM],
            settings=settings_for(server),
            show_progress=False,
        )

    assert [r.run_id for r in store.list_runs()] == ["run-1", "run-2"]
    assert len(store.query_results(run_id="run-2")) == 1


# -- two-stage sweeps ------------------------------------------------------


def answers_wrong_on(*query_ids):
    """A responder that fails exactly the named queries of `POOL`.

    Scripted replies are consumed in arrival order, which a concurrent sweep
    does not fix - so which query is answered wrongly is decided by looking at
    the request.
    """
    wrong = {f"query {query_id}?" for query_id in query_ids}

    def reply(payload):
        asked = payload["messages"][-1]["content"]
        return completion("\\boxed{5}" if asked in wrong else "\\boxed{4}")

    return reply


def two_stage(
    store,
    server,
    *,
    problems=POOL,
    approaches=("none", "bon"),
    control_fraction=0.0,
    seed=0,
    **overrides,
):
    run_two_stage_sweep(
        store,
        run_id="run-1",
        dataset="gsm8k",
        approaches=list(approaches),
        problems=list(problems),
        settings=settings_for(server, **overrides),
        control_fraction=control_fraction,
        seed=seed,
        show_progress=False,
    )


def queries_touched(store, approach):
    return {r.query_id for r in store.query_results(approach=approach)}


def test_stage_one_runs_the_baseline_across_the_whole_pool(store):
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on("2", "5")
        two_stage(store, server)

    assert queries_touched(store, "none") == {"1", "2", "3", "4", "5", "6"}


def test_stage_two_runs_only_on_what_the_baseline_got_wrong(store):
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on("2", "5")
        two_stage(store, server, control_fraction=0.0)

    assert queries_touched(store, "bon") == {"2", "5"}


def test_a_baseline_error_sends_its_query_to_stage_two(store):
    """An errored baseline is not a win, so the query still needs the field."""
    with FakeInferenceServer() as server:
        server.responder = lambda payload: (
            500 if "query 3?" in payload["messages"][-1]["content"] else completion()
        )
        two_stage(store, server, approaches=("none", "bon"), retries=0)

    assert queries_touched(store, "bon") == {"3"}


def test_the_control_sample_adds_queries_the_baseline_answered(store):
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on("2")
        two_stage(store, server, control_fraction=1.0)

    assert queries_touched(store, "bon") == {"1", "2", "3", "4", "5", "6"}


def test_the_same_seed_selects_the_same_control_sample(tmp_path):
    def sample(seed):
        with ResultsStore.open(tmp_path / f"seed-{seed}.sqlite") as store:
            with FakeInferenceServer() as server:
                server.responder = answers_wrong_on("2")
                two_stage(store, server, control_fraction=0.5, seed=seed)
            return queries_touched(store, "bon")

    first, again = sample(11), sample(11)
    assert first == again
    assert len(first) == 4  # the one failure, plus half of the five it solved


def test_both_stages_record_under_one_run_id(store):
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on("2")
        two_stage(store, server)

    runs = store.list_runs()
    assert [r.run_id for r in runs] == ["run-1"]
    assert runs[0].approaches == ["bon", "none"]


def test_every_non_baseline_approach_runs_in_stage_two(store):
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on("4")
        two_stage(store, server, approaches=("none", "bon", "moa"))

    assert queries_touched(store, "bon") == {"4"}
    assert queries_touched(store, "moa") == {"4"}


def test_the_cheap_tier_runs_on_the_whole_pool_not_just_stage_two(store):
    """A cheap approach can beat the baseline's cost even where it was correct,
    so it needs a chance to compete everywhere - not just where it was wrong."""
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on("2")
        two_stage(store, server, approaches=("none", "cot_reflection"), control_fraction=0.0)

    assert queries_touched(store, "cot_reflection") == {"1", "2", "3", "4", "5", "6"}


def test_the_expensive_tier_still_runs_only_on_what_the_baseline_got_wrong(store):
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on("2")
        two_stage(
            store,
            server,
            approaches=("none", "cot_reflection", "bon"),
            control_fraction=0.0,
        )

    assert queries_touched(store, "cot_reflection") == {"1", "2", "3", "4", "5", "6"}
    assert queries_touched(store, "bon") == {"2"}


def test_the_baseline_runs_once_even_when_it_is_not_requested(store):
    """Stage one is the baseline by definition - the label rule needs it."""
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on("4")
        two_stage(store, server, approaches=("bon",))

    assert queries_touched(store, "none") == {"1", "2", "3", "4", "5", "6"}
    assert queries_touched(store, "bon") == {"4"}


class StoreThatDies(ResultsStore):
    """A real store that stops accepting writes part-way through a sweep.

    Stands in for the process being killed mid-flight - the rows written before
    it died are really in the database file, so the test can reopen it and see
    exactly what survived.
    """

    def __init__(self, connection, crash_after: int):
        super().__init__(connection)
        self._writes_left = crash_after

    @classmethod
    def open_at(cls, path, *, crash_after: int) -> "StoreThatDies":
        return cls(sqlite3.connect(str(path), check_same_thread=False), crash_after)

    def write_result(self, result):
        if self._writes_left == 0:
            raise RuntimeError("sweep killed mid-flight")
        self._writes_left -= 1
        super().write_result(result)


def test_a_two_stage_sweep_killed_mid_flight_keeps_the_work_it_finished(db_path):
    """Same crash behaviour as a single-stage sweep: finished work stays put."""
    dying = StoreThatDies.open_at(db_path, crash_after=len(POOL) + 2)

    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on(*(p.id for p in POOL))
        with pytest.raises(RuntimeError):
            two_stage(dying, server, approaches=("none", "bon"), concurrency=1)
    dying.close()

    with ResultsStore.open(db_path) as store:
        assert len(store.query_results(approach="none")) == len(POOL)
        assert len(store.query_results(approach="bon")) == 2
        assert store.sweep_configs(run_id="run-1")[0].mode == "two-stage"


def test_resuming_the_same_run_id_skips_completed_work_and_finishes_the_rest(db_path):
    """The mid-flight crash's real point: reusing --run-id should pick up where
    it died, hitting the server only for what never finished - not redoing it."""
    dying = StoreThatDies.open_at(db_path, crash_after=len(POOL) + 2)
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on(*(p.id for p in POOL))
        with pytest.raises(RuntimeError):
            two_stage(dying, server, approaches=("none", "bon"), concurrency=1)
    dying.close()

    with ResultsStore.open(db_path) as store:
        already = len(store.query_results(approach="none")) + len(
            store.query_results(approach="bon")
        )

    with ResultsStore.open(db_path) as store, FakeInferenceServer() as server:
        server.responder = answers_wrong_on(*(p.id for p in POOL))
        two_stage(store, server, approaches=("none", "bon"), concurrency=1)
        # Only the work the crashed run never completed should have gone out
        # over the wire this time - not the whole pool again.
        assert len(server.requests) == 2 * len(POOL) - already

        assert len(store.query_results(approach="none")) == len(POOL)
        assert len(store.query_results(approach="bon")) == len(POOL)


def test_a_row_that_recorded_an_error_is_retried_on_resume(store):
    with FakeInferenceServer() as server:
        server.replies = [500]
        sweep(store, server, approaches=("none",), retries=0)

    [row] = store.query_results(approach="none")
    assert row.error is not None

    with FakeInferenceServer() as server:
        sweep(store, server, approaches=("none",))

    [row] = store.query_results(approach="none")
    assert row.error is None


def test_no_resume_redoes_everything_even_if_already_done(store):
    with FakeInferenceServer() as server:
        sweep(store, server, approaches=("none",))

    with FakeInferenceServer() as server:
        sweep(store, server, approaches=("none",), resume=False)
        assert len(server.requests) == 1


def test_the_sweep_configuration_is_recorded_for_the_run(store):
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on("2")
        two_stage(store, server, approaches=("none", "bon"), control_fraction=0.25, seed=3)

    config = store.sweep_configs(run_id="run-1")[0]
    assert config.mode == "two-stage"
    assert config.dataset == "gsm8k"
    assert config.model == "qwen3-8b"
    assert config.pool_size == 6
    assert config.approaches == ["none", "bon"]
    assert config.control_fraction == 0.25
    assert config.seed == 3


def test_a_single_stage_sweep_records_its_configuration_too(store):
    with FakeInferenceServer() as server:
        sweep(store, server, approaches=("none", "bon"), problems=POOL)

    config = store.sweep_configs(run_id="run-1")[0]
    assert config.mode == "single"
    assert config.pool_size == 6
    assert config.approaches == ["none", "bon"]
    assert (config.control_fraction, config.seed) == (None, None)


def test_stage_two_coverage_is_visible_through_the_store(store):
    with FakeInferenceServer() as server:
        server.responder = answers_wrong_on("2")
        two_stage(store, server, approaches=("none", "bon"))

    coverage = store.approach_coverage(
        run_id="run-1", dataset="gsm8k", model="qwen3-8b"
    )
    assert coverage["2"] == ["bon", "none"]
    assert coverage["1"] == ["none"]


def test_code_datasets_are_graded_by_running_their_tests(store):
    problem = Problem(
        id="1",
        question="Write add().",
        gold="assert add(1, 2) == 3",
    )

    with FakeInferenceServer() as server:
        server.default_reply = completion(
            "Sure:\n```python\ndef add(a, b):\n    return a + b\n```"
        )
        run_sweep(
            store,
            run_id="run-1",
            dataset="mbpp",
            approaches=["none"],
            problems=[problem],
            settings=settings_for(server),
            show_progress=False,
        )

    result = store.query_results()[0]
    assert result.correct is True
    assert result.predicted == "def add(a, b):\n    return a + b"
