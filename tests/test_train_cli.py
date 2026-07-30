"""Where a finetune records, decided by the command line and nothing else.

The tracker and its replay are covered in `test_jsonl_tracker.py`; what is
asserted here is the hop the cluster depends on - that `--tracker jsonl` really
does select the flat-file tracker, and where the file lands. The Aim branch is
deliberately not exercised: constructing it would create an Aim repository, which
this suite never does.
"""

import pytest

import train_router
from router_lab.training.jsonl_tracker import JsonlTracker, read_header
from router_lab.training.tracker import NullTracker


def args_for(argv):
    return train_router.parse_args(argv)


def test_aim_stays_the_default_tracker():
    """The flat file is for the cluster; a laptop run should land in Aim directly."""
    assert args_for([]).tracker == "aim"


def test_asking_for_jsonl_gets_the_flat_file_tracker(tmp_path):
    path = tmp_path / "runs" / "cluster.jsonl"
    args = args_for(
        ["--tracker", "jsonl", "--tracker-file", str(path), "--run-name", "cluster"]
    )

    tracker = train_router.build_tracker(args)
    try:
        assert isinstance(tracker, JsonlTracker)
        assert tracker.path == path
    finally:
        tracker.close()

    header = read_header(path)
    assert header.name == "cluster"
    assert header.experiment == "router-finetune"


def test_a_jsonl_run_is_named_after_the_run_when_no_file_is_given():
    args = args_for(["--tracker", "jsonl", "--run-name", "unfrozen-sweep"])

    assert train_router.tracker_file(args).name == "router-unfrozen-sweep.jsonl"
    assert (
        train_router.tracker_file(args).parent.name
        == train_router.DEFAULT_TRACKER_DIR
    )


def test_an_unnamed_jsonl_run_is_named_after_the_clock():
    args = args_for(["--tracker", "jsonl"])

    name = train_router.tracker_file(args).name
    assert name.startswith("router-") and name.endswith(".jsonl")


def test_asking_for_no_tracker_records_nothing():
    tracker = train_router.build_tracker(args_for(["--tracker", "none"]))

    assert isinstance(tracker, NullTracker)


def test_an_unknown_tracker_is_refused_by_the_parser():
    with pytest.raises(SystemExit):
        args_for(["--tracker", "sqlite"])


def test_the_test_split_is_opt_in_on_the_command_line():
    assert args_for([]).score_test is False
    assert args_for(["--score-test"]).score_test is True
