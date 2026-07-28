"""Eval-harness seam: the HTTP boundary between the harness and optillm.

Every test here points the harness at a fake OpenAI-compatible server (see
`tests.fake_server`) and asserts on what landed in a real SQLite store.
"""

import pytest

from router_lab.datasets import Problem
from router_lab.harness import SweepSettings, run_sweep
from router_lab.store import ResultsStore

from tests.fake_server import FakeInferenceServer, completion

PROBLEM = Problem(id="1", question="What is 2 + 2?", gold="4")


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


def sweep(store, server, *, approaches=("none",), problems=(PROBLEM,), **overrides):
    run_sweep(
        store,
        run_id="run-1",
        dataset="gsm8k",
        approaches=list(approaches),
        problems=list(problems),
        settings=settings_for(server, **overrides),
        show_progress=False,
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
