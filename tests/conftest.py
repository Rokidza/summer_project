"""Fixtures and factories shared across the test suite."""

import pytest

from router_lab.store import BenchmarkResult, ResultsStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "results.sqlite"


@pytest.fixture
def store(db_path):
    with ResultsStore.open(db_path) as store:
        yield store


@pytest.fixture
def make_result():
    return build_result


def build_result(**overrides) -> BenchmarkResult:
    """A fully-populated result; override only what a test cares about.

    Every field has a plausible value so a test can state just the one or two
    that matter to it and still get a row the store will accept.
    """
    fields = dict(
        run_id="run-1",
        dataset="gsm8k",
        model="qwen3-8b",
        approach="bon",
        query_id="7",
        question="What is 2 + 2?",
        correct=True,
        predicted="4",
        gold="4",
        latency_s=1.25,
        completion_tokens=128,
        total_tokens=256,
        call_count=3,
        error=None,
        router_approach=None,
        response="The answer is \\boxed{4}",
    )
    fields.update(overrides)
    return BenchmarkResult(**fields)
