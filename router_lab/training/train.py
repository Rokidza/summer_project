"""The finetuning loop: labelled examples in, a checkpoint and a tracked run out.

Deliberately thin. It trains the classification head and the effort encoder on
top of optillm's frozen sentence encoder, and records loss, agreement with the
label, and - given a table of recorded outcomes - what its predictions would
actually have achieved.

Agreement is kept as a diagnostic and named as one (`label_agreement`), because
it is a poor score: a router that picks a *different* approach which solved the
same query at the same cost fails on agreement while doing a perfect job. The
metrics a run is judged by come from `router_lab.policy` via
`router_lab.training.evaluation`, and are realised accuracy and realised cost.

What the loop does insist on is the parts that fail silently: the frozen
encoder really is frozen, the effort feature really is the constant the plugin
sends, the test split stays unscored unless a run asks for it, and provenance is
written before the first batch so a run that dies half way still says what it
was training on.

The tracker is not closed here. The caller owns the run's lifetime, because a
finetune is one phase of a longer session - evaluation follows it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from router_lab.policy import OutcomeTable
from router_lab.training.classifier import (
    Encoder,
    OptILMClassifier,
    freeze_to_head,
)
from router_lab.training.evaluation import (
    Evaluation,
    evaluate,
    log_headline,
    references_for,
)
from router_lab.training.examples import (
    TEST,
    TRAIN,
    VALIDATION,
    Example,
    ExampleSet,
    format_counts,
)
from router_lab.training.optillm_router import APPROACHES, INFERENCE_EFFORT
from router_lab.training.tracker import NullTracker, Tracker


@dataclass(frozen=True)
class TrainingConfig:
    """Everything about a finetune that is a knob rather than data."""

    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    device: str = "auto"
    seed: int = 0
    checkpoint_dir: Path = Path("checkpoints")
    run_name: str | None = None
    """Names the checkpoint file; defaults to a timestamp."""
    score_test: bool = False
    """Whether to score the held-out test split once, after training.

    Off by default and never implied: a test split scored on every run of a
    hyperparameter search is a validation split with extra steps, and stops
    being an honest estimate the first time it informs a decision.
    """

    def as_params(self) -> dict:
        return {
            "train/epochs": self.epochs,
            "train/batch_size": self.batch_size,
            "train/learning_rate": self.learning_rate,
            "train/weight_decay": self.weight_decay,
            "train/device": self.device,
            "train/seed": self.seed,
            "train/effort": INFERENCE_EFFORT,
            "train/label_space": ",".join(APPROACHES),
            "train/score_test": self.score_test,
        }


@dataclass(frozen=True)
class EpochMetrics:
    """What one pass over one subset produced."""

    epoch: int
    subset: str
    loss: float
    agreement: float
    """Share of examples whose predicted approach *is* the label's.

    A diagnostic, not the score - see `router_lab.policy` for the one that
    counts. Reported because a run whose agreement is pinned at the majority
    class is broken in a way loss alone does not show.
    """


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    history: list[EpochMetrics] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)
    """Realised outcomes per subset per epoch; empty when no outcome table was given."""


def train_router(
    examples: ExampleSet,
    *,
    model: OptILMClassifier,
    encode: Encoder,
    config: TrainingConfig = TrainingConfig(),
    tracker: Tracker | None = None,
    outcomes: OutcomeTable | None = None,
) -> TrainingResult:
    """Finetune the head on `examples`, recording through `tracker`.

    Only the train and validation splits are trained on or scored. The test
    split stays unread unless `config.score_test` asks for it.

    `outcomes` is what turns a run from "agreed with the label 62% of the time"
    into "would have answered 71% of queries correctly at 2.4x the baseline's
    cost". Without it the loop still trains and still records loss and
    agreement; it just cannot say whether the router is any good.
    """
    tracker = tracker or NullTracker()
    train_examples = examples.split(TRAIN)
    validation_examples = examples.split(VALIDATION)
    if not train_examples:
        raise ValueError(
            "no training examples: the built set has nothing in the train split"
        )

    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    trainable = freeze_to_head(model)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    tracker.log_params(
        {
            **examples.provenance.as_params(),
            **config.as_params(),
            "train/trainable_parameters": sum(p.numel() for p in trainable),
            "train/n_train": len(train_examples),
            "train/n_validation": len(validation_examples),
            "labels/train": format_counts(examples.label_counts(TRAIN)),
            "labels/validation": format_counts(examples.label_counts(VALIDATION)),
        }
    )

    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    passes = [
        _Pass(
            TRAIN,
            train_examples,
            _loader(train_examples, encode, config.batch_size, shuffle=True),
            optimizer,
        )
    ]
    if validation_examples:
        # A split with nothing in it is legitimate on a small database, and it is
        # left out of the loop rather than passed over: a zero logged for an empty
        # subset is indistinguishable from a measured one. Better a chart with no
        # validation line than a chart with a fabricated one.
        passes.append(
            _Pass(
                VALIDATION,
                validation_examples,
                _loader(validation_examples, encode, config.batch_size, shuffle=False),
                None,
            )
        )

    # The reference policies do not change as the head trains, so they are
    # computed once per split and re-logged at every step, drawing flat lines
    # across the run's charts.
    references = (
        {each.subset: references_for(outcomes, each.examples) for each in passes}
        if outcomes is not None
        else {}
    )

    history: list[EpochMetrics] = []
    evaluations: list[Evaluation] = []
    for epoch in range(1, config.epochs + 1):
        for each in passes:
            loss, agreement, predictions = _run_epoch(
                model, each.loader, criterion, each.optimizer, device=device
            )
            history.append(EpochMetrics(epoch, each.subset, loss, agreement))
            tracker.log_metric(
                "loss", loss, step=epoch, context={"subset": each.subset}
            )
            tracker.log_metric(
                "label_agreement", agreement, step=epoch, context={"subset": each.subset}
            )
            if outcomes is not None:
                evaluations.append(
                    evaluate(
                        each.examples,
                        predictions,
                        outcomes,
                        subset=each.subset,
                        epoch=epoch,
                        tracker=tracker,
                        references=references[each.subset],
                    )
                )

    test_evaluation = None
    if config.score_test and outcomes is not None:
        test_evaluation = _score_test(
            examples,
            model=model,
            encode=encode,
            criterion=criterion,
            config=config,
            device=device,
            tracker=tracker,
            outcomes=outcomes,
        )
        if test_evaluation is not None:
            evaluations.append(test_evaluation)

    if device.type == "cuda":
        # The head-only regime exists to fit a 4GB laptop GPU; recording the peak
        # turns that from a claim into a number each run can be checked against.
        tracker.log_summary(
            "peak_vram_mib", torch.cuda.max_memory_allocated(device) / 1024**2
        )
    checkpoint_path = _save_checkpoint(model, config)
    tracker.log_summary("checkpoint_path", str(checkpoint_path))

    headline = _headline(evaluations)
    if headline is not None:
        log_headline(tracker, headline, references.get(headline.subset))
    if test_evaluation is not None:
        log_headline(
            tracker,
            test_evaluation,
            references_for(outcomes, examples.split(TEST)),
            prefix="test",
        )
    return TrainingResult(
        checkpoint_path=checkpoint_path, history=history, evaluations=evaluations
    )


@dataclass(frozen=True)
class _Pass:
    """One subset, and everything needed to run and score a pass over it."""

    subset: str
    examples: list[Example]
    loader: DataLoader
    optimizer: torch.optim.Optimizer | None
    """None makes the pass an evaluation rather than a training step."""


def _headline(evaluations: Sequence[Evaluation]) -> Evaluation | None:
    """The evaluation a run is judged by: its last validation pass.

    Falls back to the train split only when there was no validation split to
    hold out, and never to the test split - a headline that silently became the
    test score the moment someone passed `--score-test` would be a trap.
    """
    for subset in (VALIDATION, TRAIN):
        scored = [each for each in evaluations if each.subset == subset]
        if scored:
            return scored[-1]
    return None


def _score_test(
    examples: ExampleSet,
    *,
    model: OptILMClassifier,
    encode: Encoder,
    criterion: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    tracker: Tracker,
    outcomes: OutcomeTable,
) -> Evaluation | None:
    """Score the held-out split once, at the end, because a run asked for it."""
    test_examples = examples.split(TEST)
    if not test_examples:
        return None
    loader = _loader(test_examples, encode, config.batch_size, shuffle=False)
    _, agreement, predictions = _run_epoch(
        model, loader, criterion, None, device=device
    )
    tracker.log_metric(
        "label_agreement", agreement, step=config.epochs, context={"subset": TEST}
    )
    return evaluate(
        test_examples,
        predictions,
        outcomes,
        subset=TEST,
        epoch=config.epochs,
        tracker=tracker,
    )


def _resolve_device(requested: str) -> torch.device:
    """"auto" takes a GPU if there is one; anything else is a demand, not a hint.

    Deliberately *not* the serving plugin's device resolution: that one answers
    to `$OPTILLM_ROUTER_DEVICE` because it is configured by whoever starts the
    server, while a finetune is configured by its own command line.
    """
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {requested!r} asks for a GPU, but torch reports no CUDA "
            f"device available. Use 'cpu', or 'auto' to take whatever is present."
        )
    return device


def _loader(
    examples: Sequence[Example], encode: Encoder, batch_size: int, *, shuffle: bool
) -> DataLoader:
    """Tokenise a split once, up front - the texts do not change between epochs.

    Each row carries its position in the split, so predictions can be put back
    in example order no matter how the loader shuffled them. Scoring a policy
    means joining predictions to queries, and a shuffled join is worse than no
    score at all: it would look plausible and mean nothing.
    """
    encoded = encode([example.text for example in examples])
    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor([e.effort for e in examples], dtype=torch.float),
        torch.tensor([e.label for e in examples], dtype=torch.long),
        torch.arange(len(examples), dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: OptILMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
) -> tuple[float, float, list[int]]:
    """One pass; `optimizer=None` makes it an evaluation pass.

    Returns the mean loss, agreement with the label, and the predicted label per
    example, in example order.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    predictions = [0] * len(loader.dataset)
    with torch.set_grad_enabled(training):
        for input_ids, attention_mask, effort, labels, positions in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            effort = effort.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask=attention_mask, effort=effort)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predicted = logits.argmax(dim=1)
            for position, prediction in zip(positions.tolist(), predicted.tolist()):
                predictions[position] = prediction
            total_loss += loss.item() * labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)
    if not total:
        return 0.0, 0.0, predictions
    return total_loss / total, total_correct / total, predictions


def _save_checkpoint(model: OptILMClassifier, config: TrainingConfig) -> Path:
    directory = Path(config.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = config.run_name or time.strftime("%Y%m%d-%H%M%S")
    path = directory / f"router-{name}.pt"
    torch.save(model.state_dict(), path)
    return path
