"""The vendored router contract, and a guard against upstream drifting from it.

The checkpoint was serialised against the architecture and constants in
`router_lab/training/optillm_router.py`. The upstream plugin is a separate
checkout that this repo does not track, so it can change under us - and every
way it could change here is silent: weights loading into mismatched layers,
training on text the plugin never sends, a label space whose indices mean
something else.

So the drift test reads the upstream *source* rather than importing it
(importing `router_plugin` pulls in every optillm approach and a torch stack),
and skips when the checkout is absent.
"""

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from router_lab.training.classifier import freeze_to_head  # noqa: E402
from router_lab.training.optillm_router import (  # noqa: E402
    APPROACHES,
    EFFORT_ENCODER_WIDTH,
    INFERENCE_EFFORT,
    MAX_LENGTH,
    build_input_text,
    label_index,
)
from tests.tiny_router import HIDDEN, tiny_classifier  # noqa: E402

UPSTREAM = (
    Path(__file__).resolve().parents[1]
    / "optillm"
    / "optillm"
    / "plugins"
    / "router_plugin.py"
)


@pytest.fixture(scope="module")
def upstream_source() -> str:
    if not UPSTREAM.exists():
        pytest.skip(f"no optillm checkout at {UPSTREAM}")
    return UPSTREAM.read_text()


def constant(source: str, name: str):
    """The value assigned to a module-level constant in `source`."""
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"upstream no longer defines {name}")


def default_of(source: str, function: str, argument: str):
    """The default value of a keyword argument of a top-level function."""
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            args = node.args.args[-len(node.args.defaults) :]
            defaults = dict(zip((a.arg for a in args), node.args.defaults))
            return ast.literal_eval(defaults[argument])
    raise AssertionError(f"upstream no longer defines {function}()")


# -- our side -------------------------------------------------------------


def test_the_label_space_keeps_optillms_order():
    assert APPROACHES[0] == "none"
    assert label_index("none") == 0
    assert label_index("bon") == 2
    assert len(APPROACHES) == len(set(APPROACHES)) == 13


def test_input_text_is_the_prompt_then_the_query():
    assert build_input_text("Be careful.", "2 + 2?") == "Be careful.\n\nUser: 2 + 2?"


def test_freezing_leaves_gradients_on_the_head_alone():
    model = tiny_classifier()

    trainable = freeze_to_head(model)

    names = {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    assert names == {
        "effort_encoder.0.weight",
        "effort_encoder.0.bias",
        "effort_encoder.2.weight",
        "effort_encoder.2.bias",
        "classifier.weight",
        "classifier.bias",
    }
    assert len(trainable) == len(names)


def test_the_head_reads_the_pooled_token_and_the_effort_feature():
    model = tiny_classifier()

    logits = model(
        input_ids=torch.ones(2, 4, dtype=torch.long),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        effort=torch.full((2,), INFERENCE_EFFORT),
    )

    assert logits.shape == (2, len(APPROACHES))
    assert model.classifier.in_features == HIDDEN + EFFORT_ENCODER_WIDTH


def test_a_checkpoint_load_carries_the_head_rather_than_leaving_it_random(tmp_path):
    """What `load_pretrained_router` relies on, without the 3GB download.

    The published checkpoint carries head weights, and `load_model` is strict -
    so a head that failed to load would raise rather than silently stay
    randomly-initialised. This asserts that mechanism on a checkpoint written
    here: same save/load path, one thousandth the encoder.
    """
    from safetensors.torch import load_model, save_model

    trained = tiny_classifier()
    with torch.no_grad():
        trained.classifier.weight.fill_(0.25)
    path = tmp_path / "model.safetensors"
    save_model(trained, str(path))

    fresh = tiny_classifier()
    assert not torch.equal(fresh.classifier.weight, trained.classifier.weight)
    load_model(fresh, str(path))

    assert torch.equal(fresh.classifier.weight, trained.classifier.weight)
    assert torch.equal(fresh.effort_encoder[0].weight, trained.effort_encoder[0].weight)


# -- the drift guard ------------------------------------------------------


def test_upstream_label_space_is_unchanged(upstream_source):
    assert constant(upstream_source, "APPROACHES") == APPROACHES


def test_upstream_tokenisation_is_unchanged(upstream_source):
    assert constant(upstream_source, "MAX_LENGTH") == MAX_LENGTH
    for setting in ("padding='max_length'", "truncation=True", "add_special_tokens=True"):
        assert setting in upstream_source


def test_upstream_input_construction_is_unchanged(upstream_source):
    """The one string that decides whether training and serving see the same text."""
    assert (
        'combined_input = f"{system_prompt}\\n\\nUser: {initial_query}"'
        in upstream_source
    )


def test_upstream_effort_default_is_the_pinned_constant(upstream_source):
    assert default_of(upstream_source, "predict_approach", "effort") == INFERENCE_EFFORT
    assert "predict_approach(router_model, input_ids, attention_mask, device)" in (
        upstream_source
    ), "the plugin now passes an effort value; the pinned constant may be stale"


def test_upstream_architecture_is_unchanged(upstream_source):
    """Layer names and widths are the serialised format, not an implementation detail."""
    for line in (
        "self.base_model = base_model",
        f"nn.Linear(1, {EFFORT_ENCODER_WIDTH})",
        f"nn.Linear({EFFORT_ENCODER_WIDTH}, {EFFORT_ENCODER_WIDTH})",
        "self.classifier = nn.Linear(base_model.config.hidden_size "
        f"+ {EFFORT_ENCODER_WIDTH}, num_labels)",
        "pooled_output = outputs.last_hidden_state[:, 0]",
        "self.effort_encoder(effort.unsqueeze(1))",
        "torch.cat((pooled_output, effort_encoded), dim=1)",
    ):
        assert line in upstream_source, line
