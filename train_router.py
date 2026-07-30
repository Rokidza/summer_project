#!/usr/bin/env python3
"""Finetune optillm's router on labels this platform manufactured.

Reads a results database, derives the winning approach per query with the label
rule, continues training optillm's published checkpoint on those labels, records
the run in the local Aim repository, and writes a checkpoint to disk.

    python train_router.py --db results.sqlite --epochs 3
    python train_router.py --db results.sqlite --dry-run     # build only
    aim up --repo .aim                                       # then look at it

On a compute node, record to a flat file instead of Aim and replay it locally
(Aim's RocksDB repository must never be created on cluster storage):

    python train_router.py --db results.sqlite --tracker jsonl

By default only the classification head and the effort encoder are trained; the
~400M parameter encoder stays frozen, which is what makes this fit on a 4GB
laptop GPU. `--regime top_layers` also trains the top encoder blocks and
`--regime full` trains everything - neither of which fits that GPU, and both of
which normally want a much lower `--learning-rate` than the head does. The
objective is class-weighted from the training split's label distribution, because
the label rule makes the baseline the winner of most queries.

`--dry-run` builds the examples, prints their provenance, and scores the four
reference policies without touching the network or a GPU - worth doing first,
because a set built from too few queries or with poor approach coverage is a
wasted training run, and because the reference scores say what the finetune has
to beat to have been worth running at all.
"""

import argparse
import os
import time
from pathlib import Path

from router_lab.policy import outcomes_for_runs, reference_scores
from router_lab.store import DB_ENV_VAR, DEFAULT_DB, ResultsStore
from router_lab.training.evaluation import keys_of
from router_lab.training.examples import TEST, VALIDATION, build_examples
from router_lab.training.regime import (
    BALANCED,
    DEFAULT_UNFROZEN_LAYERS,
    HEAD,
    REGIMES,
    WEIGHTING_SCHEMES,
)

DEFAULT_TRACKER_DIR = "runs"
"""Where JSONL runs land by default - beside the results database, git-ignored."""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        default=os.environ.get(DB_ENV_VAR, DEFAULT_DB),
        help=f"results database to build examples from "
        f"(default: ${DB_ENV_VAR} or {DEFAULT_DB})",
    )
    ap.add_argument("--run-ids", nargs="+", default=None, help="default: every run")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--model", default=None, help="served model whose results to use")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N")
    ap.add_argument(
        "--regime",
        choices=list(REGIMES),
        default=HEAD,
        help="how much of the model to train; only the default fits a 4GB GPU, "
        "and the other two normally want a far lower --learning-rate",
    )
    ap.add_argument(
        "--unfrozen-layers",
        type=int,
        default=DEFAULT_UNFROZEN_LAYERS,
        help=f"--regime top_layers: encoder blocks from the top "
        f"(default: {DEFAULT_UNFROZEN_LAYERS})",
    )
    ap.add_argument(
        "--class-weights",
        choices=list(WEIGHTING_SCHEMES),
        default=BALANCED,
        help="weight the objective by the training split's label distribution; "
        "'none' turns that off, which is how its effect gets measured",
    )
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--score-test",
        action="store_true",
        help="score the held-out test split once, after training - not by default",
    )
    ap.add_argument(
        "--tracker",
        choices=["aim", "jsonl", "none"],
        default="aim",
        help="aim writes the local repository; jsonl writes a flat file to replay "
        "later, which is what a cluster job uses",
    )
    ap.add_argument("--aim-repo", default=".aim", help="local disk only, never Lustre")
    ap.add_argument(
        "--tracker-file",
        default=None,
        help=f"--tracker jsonl target "
        f"(default: {DEFAULT_TRACKER_DIR}/<run-name>.jsonl)",
    )
    ap.add_argument("--experiment", default="router-finetune")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build the examples, print their provenance, and stop",
    )
    return ap.parse_args(argv)


def tracker_file(args) -> Path:
    """Where a JSONL run lands: named after the run, so it is findable later."""
    if args.tracker_file:
        return Path(args.tracker_file)
    stem = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    return Path(DEFAULT_TRACKER_DIR) / f"router-{stem}.jsonl"


def build_tracker(args):
    """The recording surface, chosen by configuration and nothing else.

    On a compute node this is `jsonl`: Aim's RocksDB backend must not be given a
    repository on cluster storage, so the run is recorded to a flat file and
    replayed into Aim afterwards with `replay_tracker.py`.
    """
    if args.tracker == "none":
        from router_lab.training.tracker import NullTracker

        return NullTracker()
    if args.tracker == "jsonl":
        from router_lab.training.jsonl_tracker import JsonlTracker

        return JsonlTracker(
            tracker_file(args), experiment=args.experiment, name=args.run_name
        )
    from router_lab.training.aim_tracker import AimTracker

    return AimTracker(
        repo=args.aim_repo, experiment=args.experiment, name=args.run_name
    )


def describe(built) -> str:
    provenance = built.provenance
    lines = [
        f"{provenance.n_examples} examples from runs "
        f"{', '.join(provenance.run_ids) or '(none)'}",
        f"  datasets:  {provenance.dataset_counts}",
        f"  splits:    {provenance.split_counts}",
        f"  coverage:  {provenance.coverage_fraction:.0%} of queries ran every approach",
        f"  dropped:   {provenance.n_dropped} (winner outside optillm's label space)",
        f"  labels:    {built.label_counts()}",
        f"  fingerprint {provenance.fingerprint[:12]} (seed {provenance.split_seed})",
    ]
    return "\n".join(lines)


def describe_references(outcomes, built) -> str:
    """What the finetune has to beat, on the split it will be judged on.

    Pure table lookups over results already in the database, so this costs
    milliseconds and is printed before any GPU is touched. On a set too small to
    have a validation split it falls back to the whole set, and says so rather
    than printing whole-set numbers under a validation heading.
    """
    scored = built.split(VALIDATION)
    where = f"the {VALIDATION} split"
    if not scored:
        scored, where = built.examples, "every labelled query (no validation split)"
    scores = reference_scores(outcomes, keys=keys_of(scored))
    lines = [f"reference policies on {where}:"]
    lines += [f"  {score.summary_line()}" for score in scores.values()]
    return "\n".join(lines)


def main(argv=None) -> None:
    args = parse_args(argv)
    # Resolved once: the default is a timestamp, and a run whose file is named
    # twice a second apart would report a path it never wrote to.
    args.tracker_file = str(tracker_file(args))

    with ResultsStore.open(args.db) as store:
        built = build_examples(
            store,
            run_ids=args.run_ids,
            model=args.model,
            datasets=args.datasets,
            seed=args.split_seed,
        )
        outcomes = outcomes_for_runs(
            store, run_ids=args.run_ids, datasets=args.datasets, model=args.model
        )

    print(describe(built))
    if built.examples:
        print()
        print(describe_references(outcomes, built))
    if args.dry_run:
        return
    if not built.examples:
        raise SystemExit(f"no labelled queries in {args.db} - run a sweep first")

    # Imported here rather than at the top so `--dry-run` works on an install
    # without the training extras: building examples is store, label-rule and
    # dataset work, and needs no deep-learning stack.
    from router_lab.training.classifier import (
        load_pretrained_router,
        tokenizer_encoder,
    )
    from router_lab.training.train import TrainingConfig, train_router

    print(
        f"\nloading optillm's router checkpoint "
        f"({args.device}, regime {args.regime}) ..."
    )
    model, tokenizer = load_pretrained_router()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        regime=args.regime,
        unfrozen_layers=args.unfrozen_layers,
        class_weighting=args.class_weights,
        checkpoint_dir=Path(args.checkpoint_dir),
        run_name=args.run_name,
        score_test=args.score_test,
    )
    tracker = build_tracker(args)
    try:
        result = train_router(
            built,
            model=model,
            encode=tokenizer_encoder(tokenizer),
            config=config,
            tracker=tracker,
            outcomes=outcomes,
        )
    finally:
        tracker.close()

    for metrics in result.history:
        print(
            f"epoch {metrics.epoch} {metrics.subset:<10} "
            f"loss {metrics.loss:.4f}  agreement {metrics.agreement:.3f}"
        )
    print()
    for evaluation in result.evaluations:
        print(f"epoch {evaluation.epoch} {evaluation.report()}")
    print(f"\ncheckpoint: {result.checkpoint_path}")
    if args.tracker == "aim":
        print(f"browse the run: aim up --repo {args.aim_repo}")
    elif args.tracker == "jsonl":
        print(
            f"recorded to {args.tracker_file}\n"
            f"replay it into Aim once it is on a laptop:\n"
            f"  python replay_tracker.py {args.tracker_file} "
            f"--aim-repo {args.aim_repo}"
        )
    if not args.score_test:
        held_out = built.split(TEST)
        print(
            f"({len(held_out)} test-split queries left unscored; "
            f"--score-test scores them once, when a run is finished being tuned)"
        )


if __name__ == "__main__":
    main()
