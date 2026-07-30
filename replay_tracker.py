#!/usr/bin/env python3
"""Replay a JSONL training run into the local Aim repository.

A finetune on a compute node records to a flat file rather than to Aim: Aim's
backend is a collection of RocksDB databases, and those must not be created on
cluster storage. The file comes home beside the results database, and this puts
it into Aim so a cluster run and a laptop run sit side by side in one UI,
comparable on identical metrics.

    python replay_tracker.py runs/router-20260730-1200.jsonl
    aim up --repo .aim

Replaying the same run twice is refused rather than duplicated - a duplicate run
would then be averaged into every comparison that includes it. Two things are
checked: a receipt written beside the file, and whether the destination already
holds a run replayed from this file's uid (which catches a fresh copy synced down
from the cluster, and a replay that died half way). `--force` overrides both.
"""

import argparse
from pathlib import Path

from router_lab.training.aim_tracker import DEFAULT_REPO, AimTracker
from router_lab.training.jsonl_tracker import (
    ReplayRefused,
    ensure_replayable,
    read_header,
    replay,
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path, help="JSONL runs to replay")
    ap.add_argument(
        "--aim-repo", default=DEFAULT_REPO, help="local disk only, never Lustre"
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="replay a file that has already been replayed, duplicating its run",
    )
    return ap.parse_args(argv)


def already_in_repo(repo_path: str, uid: str) -> bool:
    """Whether this run is in the destination already, by the uid replay records.

    The receipt beside the file catches the ordinary repeat, but it is a fact
    about the *file*: a re-synced copy from the cluster arrives without one. The
    uid travels inside the file, so the destination is the only place that can
    answer honestly - which is why this check lives here, in the one module
    allowed to touch Aim, rather than in `replay`.
    """
    from aim import Repo

    if not Repo.exists(repo_path):
        return False
    for run in Repo.from_path(repo_path).iter_runs():
        try:
            if run["replay/run_uid"] == uid:
                return True
        except Exception:
            continue  # a run that was never replayed has no such key
    return False


def main(argv=None) -> None:
    args = parse_args(argv)

    for path in args.files:
        header = read_header(path)
        # Both checks happen before the destination run is opened: refusing
        # afterwards would leave an empty run in the repository to explain later.
        try:
            ensure_replayable(path, force=args.force)
        except ReplayRefused as refused:
            raise SystemExit(f"{refused}\n(pass --force to replay it anyway)")
        if not args.force and already_in_repo(args.aim_repo, header.uid):
            raise SystemExit(
                f"{path} holds run {header.uid}, which {args.aim_repo} already "
                f"has (a copy of a file replayed elsewhere, or a replay that was "
                f"interrupted). Pass --force to replay it anyway."
            )

        tracker = AimTracker(
            repo=args.aim_repo,
            experiment=header.experiment,
            name=header.name,
        )
        try:
            report = replay(path, tracker, force=args.force, target=args.aim_repo)
            run_hash = tracker.run_hash
        finally:
            tracker.close()
        print(f"{report.summary_line()} -> {args.aim_repo} run {run_hash}")

    print(f"\nbrowse: aim up --repo {args.aim_repo}")


if __name__ == "__main__":
    main()
