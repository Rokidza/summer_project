"""Regimes and class weighting, as pure decisions about a model and a split.

Freezing is asserted here against the tiny router; what a *run* trained, and
what it recorded about it, is `test_training.py`'s job. The weighting is
arithmetic over counts, so it needs neither.
"""

import pytest

from router_lab.training.optillm_router import APPROACHES, label_index
from router_lab.training.regime import (
    BALANCED,
    FULL,
    HEAD,
    TOP_LAYERS,
    UNWEIGHTED,
    apply_regime,
    class_weights,
    encoder_blocks,
    format_weights,
    freeze_to_head,
)

pytest.importorskip("torch")

from tests.tiny_router import tiny_classifier  # noqa: E402

HEAD_PARAMETERS = {
    "effort_encoder.0.weight",
    "effort_encoder.0.bias",
    "effort_encoder.2.weight",
    "effort_encoder.2.bias",
    "classifier.weight",
    "classifier.bias",
}


def trainable_names(model) -> set[str]:
    return {name for name, p in model.named_parameters() if p.requires_grad}


# -- regimes --------------------------------------------------------------


def test_the_head_regime_leaves_gradients_on_the_head_alone():
    model = tiny_classifier()

    trainable = apply_regime(model, HEAD)

    assert trainable_names(model) == HEAD_PARAMETERS
    assert len(trainable) == len(HEAD_PARAMETERS)


def test_freeze_to_head_is_the_head_regime():
    """The default regime and the function that made it possible are one thing."""
    by_function, by_regime = tiny_classifier(), tiny_classifier()

    freeze_to_head(by_function)
    apply_regime(by_regime, HEAD)

    assert trainable_names(by_function) == trainable_names(by_regime)


def test_the_top_layers_regime_trains_the_head_and_the_top_blocks_only():
    model = tiny_classifier()  # two encoder blocks
    top = encoder_blocks(model)[-1]
    top_parameter_names = {
        f"base_model.layers.1.{name}" for name, _ in top.named_parameters()
    }

    apply_regime(model, TOP_LAYERS, unfrozen_layers=1)

    assert trainable_names(model) == HEAD_PARAMETERS | top_parameter_names
    assert top_parameter_names, "the block has parameters to unfreeze"
    assert not any(
        name.startswith("base_model.layers.0.") for name in trainable_names(model)
    )


def test_the_full_regime_trains_every_parameter():
    model = tiny_classifier()

    trainable = apply_regime(model, FULL)

    assert trainable_names(model) == {name for name, _ in model.named_parameters()}
    assert len(trainable) == len(list(model.parameters()))


def test_a_regime_starts_from_a_frozen_model_rather_than_the_last_ones_state():
    """Applying `head` after `full` must not leave the encoder trainable."""
    model = tiny_classifier()

    apply_regime(model, FULL)
    apply_regime(model, HEAD)

    assert trainable_names(model) == HEAD_PARAMETERS


def test_more_unfrozen_layers_than_the_encoder_has_is_refused():
    model = tiny_classifier()

    with pytest.raises(ValueError, match="outside 1..2"):
        apply_regime(model, TOP_LAYERS, unfrozen_layers=99)


def test_an_unknown_regime_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown training regime 'most-of-it'"):
        apply_regime(tiny_classifier(), "most-of-it")


# -- class imbalance ------------------------------------------------------


def test_balanced_weights_are_inverse_frequency_over_the_present_classes():
    weights = class_weights({"none": 8, "bon": 4}, scheme=BALANCED)

    assert weights[label_index("none")] == pytest.approx(12 / (2 * 8))
    assert weights[label_index("bon")] == pytest.approx(12 / (2 * 4))
    assert weights[label_index("none")] < weights[label_index("bon")]


def test_the_weights_average_to_one_over_the_split():
    """So a weighted run's loss stays on the same scale as an unweighted one's."""
    counts = {"none": 90, "bon": 7, "moa": 3}
    weights = class_weights(counts, scheme=BALANCED)

    mean = sum(n * weights[label_index(a)] for a, n in counts.items()) / sum(
        counts.values()
    )
    assert mean == pytest.approx(1.0)


def test_classes_the_split_has_none_of_keep_a_neutral_weight():
    weights = class_weights({"none": 5}, scheme=BALANCED)

    assert len(weights) == len(APPROACHES)
    assert weights[label_index("mcts")] == 1.0


def test_unweighted_asks_for_no_weighting_at_all():
    assert class_weights({"none": 8, "bon": 4}, scheme=UNWEIGHTED) is None


def test_an_empty_split_is_unweighted_rather_than_a_division_by_zero():
    assert class_weights({}, scheme=BALANCED) is None
    assert class_weights({"none": 0}, scheme=BALANCED) is None


def test_an_unknown_weighting_scheme_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown class weighting 'sqrt'"):
        class_weights({"none": 1}, scheme="sqrt")


def test_the_formatted_weighting_reports_what_the_split_produced():
    counts = {"none": 8, "bon": 4}

    formatted = format_weights(counts, class_weights(counts, scheme=BALANCED))

    assert formatted == "bon=1.5,none=0.75"
    assert format_weights(counts, None) == UNWEIGHTED
