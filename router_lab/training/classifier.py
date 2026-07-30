"""optillm's router architecture, vendored - the fourth thing training and
serving must agree on (see `optillm_router.py` for the other three).

This class definition is the format the published checkpoint was serialised
against. Attribute names and layer widths are part of that format, not an
implementation detail: renaming `base_model`, `effort_encoder` or `classifier`
loads weights into nothing and trains a randomly-initialised head while
reporting nothing wrong. So it is defined here rather than imported from the
`optillm/` checkout, and the drift test asserts upstream still matches.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Callable, Protocol, Sequence

import torch
import torch.nn as nn

from router_lab.training.optillm_router import (
    APPROACHES,
    BASE_MODEL,
    CHECKPOINT_REPO,
    EFFORT_ENCODER_WIDTH,
    MAX_LENGTH,
)


class OptILMClassifier(nn.Module):
    """optillm's router head over a sentence encoder. A copy of the upstream class."""

    def __init__(self, base_model, num_labels: int):
        super().__init__()
        self.base_model = base_model
        self.effort_encoder = nn.Sequential(
            nn.Linear(1, EFFORT_ENCODER_WIDTH),
            nn.ReLU(),
            nn.Linear(EFFORT_ENCODER_WIDTH, EFFORT_ENCODER_WIDTH),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(
            base_model.config.hidden_size + EFFORT_ENCODER_WIDTH, num_labels
        )

    def forward(self, input_ids, attention_mask, effort):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        effort_encoded = self.effort_encoder(effort.unsqueeze(1))
        combined_input = torch.cat((pooled_output, effort_encoded), dim=1)
        return self.classifier(combined_input)


class Encoder(Protocol):
    """Text in, `input_ids` and `attention_mask` out.

    The seam between training and HuggingFace. Production passes
    `tokenizer_encoder(tokenizer)`; tests pass something tiny and offline, which
    is what lets the loop be exercised end to end with no network and no
    downloaded weights.
    """

    def __call__(self, texts: Sequence[str]) -> dict: ...


def tokenizer_encoder(tokenizer, max_length: int = MAX_LENGTH) -> Encoder:
    """The plugin's tokenisation - same max length, padding and truncation."""

    def encode(texts: Sequence[str]) -> dict:
        encoding = tokenizer(
            list(texts),
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    return encode


def freeze_to_head(model: OptILMClassifier) -> list[nn.Parameter]:
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
    return [p for p in model.parameters() if p.requires_grad]


def load_pretrained_router(
    *, cache_dir: str | Path | None = None
) -> tuple[OptILMClassifier, Callable]:
    """optillm's published checkpoint, with its tokenizer. Touches the network.

    The base encoder is built from its config and then *overwritten* by the
    finetuned safetensors, head included: `load_model` is strict, so a
    checkpoint that did not carry head weights would raise rather than leave a
    randomly-initialised head in place. That the head loads at all is only sound
    because `APPROACHES` keeps optillm's ordering.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_model
    from transformers import AutoModel, AutoTokenizer

    base_model = AutoModel.from_pretrained(BASE_MODEL, cache_dir=cache_dir)
    model = OptILMClassifier(base_model, num_labels=len(APPROACHES))
    safetensors_path = hf_hub_download(
        repo_id=CHECKPOINT_REPO, filename="model.safetensors", cache_dir=cache_dir
    )
    load_model(model, safetensors_path)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_REPO, cache_dir=cache_dir)
    return model, tokenizer
