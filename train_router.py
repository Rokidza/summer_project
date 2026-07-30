#!/usr/bin/env python3
"""Finetune optillm's router on labels this platform manufactured.

Reads a results database, derives the winning approach per query with the label
rule, continues training optillm's published checkpoint on those labels, records
the run in the local Aim repository, and writes a checkpoint to disk.

    python train_router.py --db results.sqlite --epochs 3
    python train_router.py --db results.sqlite --dry-run     # build only
    aim up --repo .aim                                       # then look at it

Only the classification head and the effort encoder are trained; the ~400M
parameter encoder stays frozen, which is what makes this fit on a 4GB laptop
GPU. Which regime to train under becomes a choice in #18.

`--dry-run` builds the examples, prints their provenance, and scores the four
reference policies without touching the network or a GPU - worth doing first,
because a set built from too few queries or with poor approach coverage is a
wasted training run, and because the reference scores say what the finetune has
to beat to have been worth running at all.
"""

import argparse
import os
from pathlib import Path

from router_lab.policy import outcomes_for_runs, reference_scores
from router_lab.store import DB_ENV_VAR, DEFAULT_DB, ResultsStore
from router_lab.training.evaluation import keys_of
from router_lab.training.examples import TEST, VALIDATION, build_examples


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        default=os.environ.get(DB_ENV_VAR, DEFAULT_DB),
        help=f"results database to build examples from (default: ${DB_ENV_VAR} or {DEFAULT_DB})",
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
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--score-test",
        action="store_true",
        help="score the held-out test split once, after training - not by default",
    )
    ap.add_argument("--tracker", choices=["aim", "none"], default="aim")
    ap.add_argument("--aim-repo", default=".aim", help="local disk only, never Lustre")
    ap.add_argument("--experiment", default="router-finetune")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build the examples, print their provenance, and stop",
    )
    return ap.parse_args(argv)


def build_tracker(args):
    if args.tracker == "none":
        from router_lab.training.tracker import NullTracker

        return NullTracker()
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
    milliseconds and is printed before any GPU is touched.
    """
    validation = built.split(VALIDATION) or built.examples
    scores = reference_scores(outcomes, keys=keys_of(validation))
    lines = [f"reference policies on the {VALIDATION} split:"]
    lines += [f"  {score.summary_line()}" for score in scores.values()]
    return "\n".join(lines)


def main(argv=None) -> None:
    args = parse_args(argv)

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

    print(f"\nloading optillm's router checkpoint ({args.device}) ...")
    model, tokenizer = load_pretrained_router()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
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
    if not args.score_test:
        held_out = built.split(TEST)
        print(
            f"({len(held_out)} test-split queries left unscored; "
            f"--score-test scores them once, when a run is finished being tuned)"
        )


if __name__ == "__main__":
    main()
