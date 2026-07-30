"""How much of the router to train, and how to weight the classes while doing it.

Two knobs, kept together because they are the two answers to the same awkward
fact: a ~400M-parameter encoder is being finetuned on a few thousand labels of
which a large majority are one class.

- **The regime** decides which parameters receive gradients. Head-only is the
  default and the only one that fits a 4GB laptop GPU; the other two exist so
  "how much of this model should I actually train?" becomes three comparable
  runs rather than a guess. Which one suits the data is a measurement, not a
  decision to make in advance - though head-only is expected to win at low label
  volume, because fully finetuning 400M parameters on a few thousand
  heavily-imbalanced examples will memorise them inside an epoch.
- **The weighting** addresses the imbalance that survives two-stage sweeping.
  It is structural rather than incidental: the cheapest-correct label rule makes
  the baseline the winner on every query it answers correctly, so most labels are
  `none` no matter how the sweep is run.

Deliberately torch-free, like the rest of this package's non-training modules:
these names are argparse choices, so `train_router.py` reads them while parsing
a `--dry-run` command line on an install with no deep-learning stack. Nothing
here needs a torch API - freezing parameters and reading module attributes are
plain attribute access on objects the caller already has.
"""

from __future__ import annotations

from itertools import chain
from typing import Mapping, Sequence

from router_lab.training.optillm_router import APPROACHES, label_index

# -- regimes --------------------------------------------------------------

HEAD = "head"
"""Classification head and effort encoder only - the default, and the 4GB one."""

TOP_LAYERS = "top_layers"
"""The head, plus the top N encoder blocks."""

FULL = "full"
"""Everything, encoder included. Will not fit a laptop GPU - that is #21's job."""

REGIMES = (HEAD, TOP_LAYERS, FULL)

DEFAULT_UNFROZEN_LAYERS = 4
"""Blocks unfrozen by `top_layers` when a run does not say. ModernBERT-large has
28, so this is the top seventh of the encoder - enough to adapt its
representation without paying for the whole stack's optimizer state."""


def freeze_to_head(model) -> list:
    """Leave gradients on for the classification head and effort encoder only.

    This is what makes the finetune fit in 4GB: the ~400M-parameter encoder
    keeps no optimizer state and accumulates no gradients, so only activations
    scale with batch size.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in chain(
        model.effort_encoder.parameters(), model.classifier.parameters()
    ):
        parameter.requires_grad_(True)
    return trainable_parameters(model)


def encoder_blocks(model) -> list:
    """The encoder's transformer blocks, bottom first.

    Read off the attribute ModernBERT actually uses rather than by walking the
    module tree looking for something block-shaped: a wrong guess would unfreeze
    the wrong parameters and still report the regime it was asked for, which is
    the one failure mode this module must not have.
    """
    blocks = getattr(model.base_model, "layers", None)
    if blocks is None:
        raise TypeError(
            f"{type(model.base_model).__name__} exposes no `layers`, so the "
            f"{TOP_LAYERS!r} regime cannot tell which blocks are the top ones. "
            f"The router's encoder is ModernBERT, which does."
        )
    return list(blocks)


def top_layers(model, count: int) -> list:
    """The top `count` encoder blocks, refusing a count the encoder cannot honour."""
    blocks = encoder_blocks(model)
    if not 1 <= count <= len(blocks):
        raise ValueError(
            f"{count} unfrozen layers is outside 1..{len(blocks)}: the encoder "
            f"has {len(blocks)} blocks. Use {FULL!r} to train all of it."
        )
    return blocks[-count:]


def apply_regime(
    model,
    regime: str = HEAD,
    *,
    unfrozen_layers: int = DEFAULT_UNFROZEN_LAYERS,
) -> list:
    """Set `requires_grad` to exactly the parameter set `regime` names.

    Returns the parameters that will be trained, which is what the optimizer is
    built from and what the run records as `train/trainable_parameters`. Each
    regime starts from a fully frozen model rather than from whatever the last
    call left behind, so applying one twice - or after another - is the same as
    applying it once.
    """
    if regime not in REGIMES:
        raise ValueError(
            f"unknown training regime {regime!r}: one of {', '.join(REGIMES)}"
        )

    freeze_to_head(model)
    if regime == TOP_LAYERS:
        for block in top_layers(model, unfrozen_layers):
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    elif regime == FULL:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return trainable_parameters(model)


def trainable_parameters(model) -> list:
    return [p for p in model.parameters() if p.requires_grad]


# -- class imbalance ------------------------------------------------------

BALANCED = "balanced"
"""Inverse-frequency weights over the classes the training split actually has."""

UNWEIGHTED = "none"
"""No weighting - kept selectable so the effect of weighting is measurable."""

WEIGHTING_SCHEMES = (BALANCED, UNWEIGHTED)


def class_weights(
    counts: Mapping[str, int], *, scheme: str = BALANCED
) -> list[float] | None:
    """Per-class loss weights over optillm's label space, from a split's counts.

    `None` means "unweighted", which is what a criterion wants to be handed.

    Under `balanced`, a class with n of the split's N examples over K present
    classes gets N / (K * n) - so the weights average to 1.0 across the split
    and a weighted run's loss curve stays on the same scale as an unweighted
    one's. Two runs differing only in the weighting are then readable on one
    chart instead of one being an order of magnitude above the other.

    Classes the split has none of keep weight 1.0. They are never a target, so
    the value cannot reach the loss; 1.0 rather than 0.0 because a zero here
    would read as a deliberate instruction to ignore a class.
    """
    if scheme == UNWEIGHTED:
        return None
    if scheme != BALANCED:
        raise ValueError(
            f"unknown class weighting {scheme!r}: one of {', '.join(WEIGHTING_SCHEMES)}"
        )

    present = {approach: n for approach, n in counts.items() if n}
    if not present:
        return None
    total = sum(present.values())
    weights = [1.0] * len(APPROACHES)
    for approach, n in present.items():
        weights[label_index(approach)] = total / (len(present) * n)
    return weights


def format_weights(
    counts: Mapping[str, int], weights: Sequence[float] | None
) -> str:
    """The weighting as it was actually applied, for the run's parameters.

    The scheme's name says what was asked for; this says what came out of the
    split, which is the part that explains a run.
    """
    if weights is None:
        return UNWEIGHTED
    return ",".join(
        f"{approach}={weights[label_index(approach)]:.3g}"
        for approach in sorted(counts)
        if counts[approach]
    )
