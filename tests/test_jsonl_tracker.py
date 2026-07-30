"""Recording a run to a flat file, and replaying it into a tracker.

Nothing here imports Aim or creates an Aim repository - that is the point of the
protocol. What replay owes its caller is a sequence of tracker calls, so the
assertions are made against a recording tracker: the same file replayed must
produce the same calls a directly-tracked run made.
"""

import json

import pytest

from router_lab.training.jsonl_tracker import (
    SCHEMA_VERSION,
    JsonlTracker,
    ReplayRefused,
    ensure_replayable,
    read_header,
    replay,
)
from router_lab.training.tracker import Tracker
from tests.recording_tracker import RecordingTracker


class FakeFigure:
    """Stands in for a plotly figure: the one method serialisation needs."""

    def __init__(self, payload):
        self.payload = payload

    def to_plotly_json(self):
        return self.payload


def record_a_run(tracker) -> None:
    """Every kind of call a finetune makes, against any Tracker implementation."""
    tracker.log_params({"train/epochs": 2, "examples/fingerprint": "deadbeef"})
    tracker.log_metric("loss", 1.5, step=1, context={"subset": "train"})
    tracker.log_metric("loss", 0.5, step=2, context={"subset": "train"})
    tracker.log_metric(
        "realised_accuracy",
        0.75,
        step=2,
        context={"subset": "validation", "policy": "model"},
    )
    tracker.log_metric("no_context_no_step", 3.0)
    tracker.log_text(
        "misroutes", "gsm8k/7 predicted bon", step=2, context={"subset": "train"}
    )
    tracker.log_figure("confusion", FakeFigure({"data": [], "layout": {}}), step=2)
    tracker.log_summary("checkpoint_path", "checkpoints/router-test.pt")
    tracker.log_summary("final/realised_accuracy", 0.75)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "runs" / "cluster-run.jsonl"


def lines_of(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_run_is_recorded_as_one_json_object_per_call(path):
    tracker = JsonlTracker(path, experiment="router-finetune", name="cluster-run")
    record_a_run(tracker)
    tracker.close()

    records = lines_of(path)
    assert records[0]["record"] == "run"
    assert records[0]["schema"] == SCHEMA_VERSION
    assert records[0]["experiment"] == "router-finetune"
    assert records[0]["name"] == "cluster-run"
    assert [r["record"] for r in records[1:]] == [
        "params",
        "metric",
        "metric",
        "metric",
        "metric",
        "text",
        "figure",
        "summary",
        "summary",
        "end",
    ]


def test_the_file_is_created_with_its_parent_directory(path):
    JsonlTracker(path).close()

    assert path.exists()


def test_replaying_reproduces_a_directly_tracked_run(path):
    direct = RecordingTracker()
    record_a_run(direct)

    written = JsonlTracker(path, name="cluster-run")
    record_a_run(written)
    written.close()

    replayed = RecordingTracker()
    replay(path, replayed)

    assert replayed.params == direct.params
    assert replayed.metrics == direct.metrics
    assert replayed.texts == direct.texts
    # Everything but the provenance replay adds, so a replayed run can still say
    # where it came from.
    assert {
        key: value
        for key, value in replayed.summaries.items()
        if not key.startswith("replay/")
    } == direct.summaries
    assert [call.name for call in replayed.figures] == ["confusion"]
    assert replayed.figures[0].value == {"data": [], "layout": {}}
    assert replayed.figures[0].step == 2


def test_replay_says_where_the_run_came_from(path):
    written = JsonlTracker(path, name="cluster-run")
    written.log_params({"train/epochs": 1})
    written.close()
    tracker = RecordingTracker()

    report = replay(path, tracker)

    assert tracker.summaries["replay/source"] == str(path)
    assert tracker.summaries["replay/run_uid"] == report.uid == read_header(path).uid


def test_replay_does_not_close_the_run_it_writes_into(path):
    """The caller owns the destination run's lifetime, as everywhere else."""
    JsonlTracker(path).close()
    tracker = RecordingTracker()

    replay(path, tracker)

    assert tracker.closed is False


def test_replaying_twice_is_refused_rather_than_duplicated(path):
    written = JsonlTracker(path, name="cluster-run")
    written.log_metric("loss", 1.0, step=1)
    written.close()
    replay(path, RecordingTracker())

    with pytest.raises(ReplayRefused, match="already replayed"):
        replay(path, RecordingTracker())


def test_a_refused_replay_can_be_forced(path):
    JsonlTracker(path).close()
    replay(path, RecordingTracker())

    again = RecordingTracker()
    report = replay(path, again, force=True)

    assert report.uid == read_header(path).uid


def test_the_receipt_records_what_was_replayed_where(path):
    JsonlTracker(path, name="cluster-run").close()

    report = replay(path, RecordingTracker(), target=".aim")

    receipt = json.loads(report.receipt_path.read_text())
    assert receipt["uid"] == report.uid
    assert receipt["target"] == ".aim"
    assert receipt["source"] == str(path)


def test_a_run_killed_mid_epoch_is_still_replayable(path):
    """A compute node's job can be cut off; the file has to survive that.

    Every call is flushed as it is made, and the closing record is a nicety
    rather than a requirement - so a walltime kill costs the last epoch, not
    the run.
    """
    written = JsonlTracker(path, name="cluster-run")
    written.log_params({"train/epochs": 3})
    written.log_metric("loss", 1.0, step=1, context={"subset": "train"})
    # No close(): the job was killed here.
    tracker = RecordingTracker()

    replay(path, tracker)

    assert tracker.params == {"train/epochs": 3}
    assert tracker.metric_values("loss", subset="train") == [1.0]


def test_a_figure_that_cannot_be_serialised_is_refused_when_it_is_logged(path):
    tracker = JsonlTracker(path)

    with pytest.raises(TypeError, match="figure"):
        tracker.log_figure("confusion", object())


def test_an_awkward_value_is_recorded_rather_than_crashing_the_run(path, tmp_path):
    """A run that dies on its last line because a value was a Path is a bad trade."""
    tracker = JsonlTracker(path)
    tracker.log_summary("checkpoint_path", tmp_path / "router.pt")
    tracker.close()

    replayed = RecordingTracker()
    replay(path, replayed)

    assert replayed.summaries["checkpoint_path"] == str(tmp_path / "router.pt")


def test_close_is_safe_twice(path):
    tracker = JsonlTracker(path)
    tracker.close()
    tracker.close()

    assert [r["record"] for r in lines_of(path)].count("end") == 1


def test_a_file_that_does_not_start_with_a_run_header_is_refused(path):
    path.parent.mkdir(parents=True)
    path.write_text('{"record": "metric", "name": "loss", "value": 1.0}\n')

    with pytest.raises(ValueError, match="run header"):
        replay(path, RecordingTracker())


def test_a_corrupt_line_names_itself(path):
    written = JsonlTracker(path)
    written.close()
    with path.open("a") as handle:
        handle.write("{not json}\n")

    with pytest.raises(ValueError, match="line 3"):
        replay(path, RecordingTracker())


def test_an_unknown_record_kind_is_refused(path):
    written = JsonlTracker(path)
    written.close()
    with path.open("a") as handle:
        handle.write('{"record": "prophecy"}\n')

    with pytest.raises(ValueError, match="prophecy"):
        replay(path, RecordingTracker())


def test_an_empty_file_is_refused(path):
    path.parent.mkdir(parents=True)
    path.write_text("")

    with pytest.raises(ValueError, match="empty"):
        read_header(path)


def test_it_is_a_tracker(path):
    """The loop is written against the protocol, so this has to satisfy it."""
    assert isinstance(JsonlTracker(path), Tracker)


def test_a_replayed_file_can_be_checked_before_a_destination_run_is_opened(path):
    """Refusing halfway would leave an empty run in the repository to explain."""
    JsonlTracker(path).close()
    ensure_replayable(path)  # not replayed yet: no complaint
    replay(path, RecordingTracker())

    with pytest.raises(ReplayRefused):
        ensure_replayable(path)
    ensure_replayable(path, force=True)
