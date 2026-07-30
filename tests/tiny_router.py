"""A router model small enough to train in a test, built entirely offline.

The real encoder is ModernBERT-large behind a Hub download. This is the same
architecture from a config written here - a few thousand parameters, random
weights, no network - so the training loop can be exercised end to end without
anything downloaded. It says nothing about convergence and is not meant to.
"""

from transformers import AutoModel, ModernBertConfig

from router_lab.training.classifier import OptILMClassifier
from router_lab.training.optillm_router import APPROACHES

VOCAB = 64
HIDDEN = 32


def tiny_config() -> ModernBertConfig:
    return ModernBertConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )


def tiny_classifier() -> OptILMClassifier:
    return OptILMClassifier(
        AutoModel.from_config(tiny_config()), num_labels=len(APPROACHES)
    )
