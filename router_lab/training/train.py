"""The finetuning loop: labelled examples in, a checkpoint and a tracked run out.

Deliberately thin. It trains the classification head and the effort encoder on
top of optillm's frozen sentence encoder, and records loss and raw agreement
with the label. Agreement is a placeholder metric and known to be a poor one -
a router that picks a *different* approach which solved the same query at the
same cost scores as a failure here. #17 replaces it with realised outcomes.

What the loop does insist on is the parts that fail silently: the frozen
encoder really is frozen, the effort feature really is the constant the plugin
sends, the test split is never touched, and provenance is written before the
first batch so a run that dies half way still says what it was training on.

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

from router_lab.training.classifier import (
    Encoder,
    OptILMClassifier,
    freeze_to_head,
)
from router_lab.training.examples import (
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
        }


@dataclass(frozen=True)
class EpochMetrics:
    """What one pass over one subset produced."""

    epoch: int
    subset: str
    loss: float
    accuracy: float


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    history: list[EpochMetrics] = field(default_factory=list)


def train_router(
    examples: ExampleSet,
    *,
    model: OptILMClassifier,
    encode: Encoder,
    config: TrainingConfig = TrainingConfig(),
    tracker: Tracker | None = None,
) -> TrainingResult:
    """Finetune the head on `examples`, recording through `tracker`.

    Only the train and validation splits are read. The test split exists so it
    can stay unread until a run is finished being tuned.
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
        (TRAIN, _loader(train_examples, encode, config.batch_size, shuffle=True), optimizer)
    ]
    if validation_examples:
        # A split with nothing in it is legitimate on a small database, and it is
        # left out of the loop rather than passed over: a zero logged for an empty
        # subset is indistinguishable from a measured one. Better a chart with no
        # validation line than a chart with a fabricated one.
        passes.append(
            (
                VALIDATION,
                _loader(validation_examples, encode, config.batch_size, shuffle=False),
                None,
            )
        )

    history: list[EpochMetrics] = []
    for epoch in range(1, config.epochs + 1):
        for subset, loader, subset_optimizer in passes:
            loss, accuracy = _run_epoch(
                model, loader, criterion, subset_optimizer, device=device
            )
            history.append(EpochMetrics(epoch, subset, loss, accuracy))
            tracker.log_metric("loss", loss, step=epoch, context={"subset": subset})
            tracker.log_metric(
                "accuracy", accuracy, step=epoch, context={"subset": subset}
            )

    if device.type == "cuda":
        # The head-only regime exists to fit a 4GB laptop GPU; recording the peak
        # turns that from a claim into a number each run can be checked against.
        tracker.log_summary(
            "peak_vram_mib", torch.cuda.max_memory_allocated(device) / 1024**2
        )
    checkpoint_path = _save_checkpoint(model, config)
    tracker.log_summary("checkpoint_path", str(checkpoint_path))
    return TrainingResult(checkpoint_path=checkpoint_path, history=history)


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
    """Tokenise a split once, up front - the texts do not change between epochs."""
    encoded = encode([example.text for example in examples])
    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor([e.effort for e in examples], dtype=torch.float),
        torch.tensor([e.label for e in examples], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: OptILMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
) -> tuple[float, float]:
    """One pass; `optimizer=None` makes it an evaluation pass."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.set_grad_enabled(training):
        for input_ids, attention_mask, effort, labels in loader:
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

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    if not total:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def _save_checkpoint(model: OptILMClassifier, config: TrainingConfig) -> Path:
    directory = Path(config.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = config.run_name or time.strftime("%Y%m%d-%H%M%S")
    path = directory / f"router-{name}.pt"
    torch.save(model.state_dict(), path)
    return path
