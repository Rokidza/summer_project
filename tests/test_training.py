"""The training loop, exercised end to end without a network or a real router.

The encoder is a randomly-initialised ModernBERT built from a local config (see
`tests/tiny_router.py`), so nothing is downloaded; the tokenizer is replaced by
a trivial encoder at the seam the loop takes it through. Wiring is asserted -
convergence is not.
"""

from dataclasses import replace

import pytest

from router_lab.policy import ALWAYS_BASELINE, ORACLE

torch = pytest.importorskip("torch")

from router_lab.training.examples import (  # noqa: E402
    TEST,
    TRAIN,
    VALIDATION,
    Example,
    ExampleSet,
    Provenance,
)
from router_lab.training.optillm_router import (  # noqa: E402
    APPROACHES,
    INFERENCE_EFFORT,
)
from router_lab.training.regime import (  # noqa: E402
    BALANCED,
    FULL,
    HEAD,
    TOP_LAYERS,
    UNWEIGHTED,
)
from router_lab.training.train import (  # noqa: E402
    TrainingConfig,
    _loader,
    _run_epoch,
    train_router,
)
from tests.conftest import outcome_table, query_rows  # noqa: E402
from tests.recording_tracker import RecordingTracker  # noqa: E402
from tests.tiny_router import VOCAB, tiny_classifier  # noqa: E402

SEQ_LEN = 8


@pytest.fixture
def model():
    return tiny_classifier()


@pytest.fixture
def encode():
    """Bytes into ids, padded to a fixed length - the tokenizer's shape, not its work."""

    def encoder(texts):
        ids = torch.zeros(len(texts), SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(len(texts), SEQ_LEN, dtype=torch.long)
        for row, text in enumerate(texts):
            for column, char in enumerate(text.encode()[:SEQ_LEN]):
                ids[row, column] = char % VOCAB
                mask[row, column] = 1
        return {"input_ids": ids, "attention_mask": mask}

    return encoder


def make_examples(n=12) -> ExampleSet:
    """A set spanning all three splits, with two classes."""
    splits = [TRAIN] * (n - 4) + [VALIDATION] * 2 + [TEST] * 2
    approaches = ["none", "bon"]
    examples = [
        Example(
            query_id=str(i),
            dataset="gsm8k",
            text=f"Solve this problem number {i}",
            approach=approaches[i % 2],
            label=APPROACHES.index(approaches[i % 2]),
            split=splits[i],
        )
        for i in range(n)
    ]
    provenance = Provenance(
        run_ids=["run-1"],
        models=["qwen3-8b"],
        dataset_counts={"gsm8k": n},
        n_examples=n,
        n_queries=n,
        n_dropped=0,
        coverage_fraction=1.0,
        split_seed=0,
        split_ratios=(0.7, 0.15, 0.15),
        split_counts={TRAIN: n - 4, VALIDATION: 2, TEST: 2},
        n_unsolved=0,
        label_rule_version="1",
        fingerprint="deadbeef",
    )
    return ExampleSet(examples=examples, provenance=provenance)


@pytest.fixture
def config(tmp_path):
    return TrainingConfig(
        epochs=2,
        batch_size=4,
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        run_name="test-run",
    )


def test_the_loop_runs_end_to_end_and_closes_the_tracker(model, encode, config):
    tracker = RecordingTracker()

    result = train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker
    )

    assert result.checkpoint_path.exists()
    assert tracker.closed is False, "closing the run is the caller's business"
    assert len(result.history) == 2 * 2  # two epochs, two subsets


def test_only_the_head_and_effort_encoder_receive_gradients(model, encode, config):
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    train_router(make_examples(), model=model, encode=encode, config=config)

    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    }
    assert changed, "nothing trained at all"
    assert all(
        name.startswith(("classifier.", "effort_encoder.")) for name in changed
    ), sorted(changed)
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("base_model.")
    )


def test_the_top_layers_regime_trains_the_top_block_as_well(model, encode, config):
    """What a regime *trained* is read off the gradients, not off its name."""
    train_router(
        make_examples(),
        model=model,
        encode=encode,
        config=replace(config, regime=TOP_LAYERS, unfrozen_layers=1),
    )

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any(name.startswith("base_model.layers.1.") for name in trainable)
    assert not any(name.startswith("base_model.layers.0.") for name in trainable)


def test_the_full_regime_trains_the_encoder_too(model, encode, config):
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    train_router(
        make_examples(), model=model, encode=encode, config=replace(config, regime=FULL)
    )

    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    }
    assert any(name.startswith("base_model.layers.0.") for name in changed), sorted(
        changed
    )


def test_the_regime_and_its_trainable_count_are_recorded_as_parameters(
    model, encode, config
):
    """Three runs differing only by regime have to be distinguishable in the UI."""
    head, full = RecordingTracker(), RecordingTracker()

    train_router(make_examples(), model=model, encode=encode, config=config, tracker=head)
    train_router(
        make_examples(),
        model=tiny_classifier(),
        encode=encode,
        config=replace(config, regime=FULL),
        tracker=full,
    )

    assert head.params["train/regime"] == HEAD
    assert full.params["train/regime"] == FULL
    assert (
        full.params["train/trainable_parameters"]
        > head.params["train/trainable_parameters"]
    )


def test_the_objective_is_weighted_from_the_training_splits_distribution(
    model, encode, config
):
    tracker = RecordingTracker()

    train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker
    )

    # make_examples() alternates the two classes, so a balanced weighting of its
    # train split is the neutral one - and says so rather than saying nothing.
    assert tracker.params["train/class_weighting"] == BALANCED
    assert tracker.params["labels/class_weights"] == "bon=1,none=1"


def test_a_skewed_split_is_weighted_toward_the_rare_class(model, encode, config):
    built = make_examples()
    skewed = ExampleSet(
        examples=[
            e if i % 4 else replace(e, approach="bon", label=APPROACHES.index("bon"))
            for i, e in enumerate(built.examples)
        ],
        provenance=built.provenance,
    )
    tracker = RecordingTracker()

    train_router(
        skewed, model=model, encode=encode, config=config, tracker=tracker
    )

    weights = dict(
        pair.split("=") for pair in tracker.params["labels/class_weights"].split(",")
    )
    assert float(weights["none"]) > float(weights["bon"]) > 0


def test_weighting_can_be_turned_off_so_its_effect_is_measurable(
    model, encode, config
):
    tracker = RecordingTracker()

    train_router(
        make_examples(),
        model=model,
        encode=encode,
        config=replace(config, class_weighting=UNWEIGHTED),
        tracker=tracker,
    )

    assert tracker.params["train/class_weighting"] == UNWEIGHTED
    assert tracker.params["labels/class_weights"] == UNWEIGHTED


def test_records_per_epoch_loss_and_agreement_under_a_subset_context(
    model, encode, config
):
    tracker = RecordingTracker()

    train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker
    )

    assert len(tracker.metric_values("loss", subset=TRAIN)) == 2
    assert len(tracker.metric_values("label_agreement", subset=TRAIN)) == 2
    assert len(tracker.metric_values("loss", subset=VALIDATION)) == 2
    assert [call.step for call in tracker.metrics if call.name == "loss"] == [1, 1, 2, 2]
    assert all(
        0.0 <= value <= 1.0
        for value in tracker.metric_values("label_agreement", subset=TRAIN)
    )


def test_the_test_split_is_never_scored(model, encode, config):
    tracker = RecordingTracker()

    train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker
    )

    assert {TEST} not in [
        set(c.values()) for c in tracker.contexts_of("label_agreement")
    ]


def test_an_empty_validation_split_is_left_unlogged_not_logged_as_zero(
    model, encode, config
):
    """A logged zero for a subset with nothing in it reads as a measurement."""
    built = make_examples()
    train_only = ExampleSet(
        examples=[e for e in built.examples if e.split == TRAIN],
        provenance=built.provenance,
    )
    tracker = RecordingTracker()

    train_router(
        train_only, model=model, encode=encode, config=config, tracker=tracker
    )

    assert tracker.metric_values("loss", subset=TRAIN)
    assert tracker.metric_values("loss", subset=VALIDATION) == []
    assert tracker.params["train/n_validation"] == 0


def test_records_provenance_and_configuration_as_parameters(model, encode, config):
    tracker = RecordingTracker()

    train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker
    )

    assert tracker.params["examples/fingerprint"] == "deadbeef"
    assert tracker.params["examples/run_ids"] == "run-1"
    assert tracker.params["examples/label_rule_version"] == "1"
    assert tracker.params["examples/coverage_fraction"] == 1.0
    assert tracker.params["train/epochs"] == 2
    assert tracker.params["train/effort"] == INFERENCE_EFFORT
    assert tracker.params["train/label_space"] == ",".join(APPROACHES)
    assert tracker.params["train/trainable_parameters"] > 0
    assert tracker.params["labels/train"] == "bon=4,none=4"


def test_the_checkpoint_path_is_recorded_on_the_run(model, encode, config):
    tracker = RecordingTracker()

    result = train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker
    )

    assert tracker.summaries["checkpoint_path"] == str(result.checkpoint_path)
    reloaded = torch.load(result.checkpoint_path, weights_only=True)
    assert "classifier.weight" in reloaded


def test_a_run_with_no_training_examples_is_refused(model, encode, config):
    empty = ExampleSet(examples=[], provenance=make_examples().provenance)

    with pytest.raises(ValueError, match="no training examples"):
        train_router(empty, model=model, encode=encode, config=config)


def test_the_same_seed_trains_the_same_head(encode, config, model):
    """Determinism is what makes two runs in the Aim UI comparable."""
    tracker_a, tracker_b = RecordingTracker(), RecordingTracker()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker_a
    )
    first = tracker_a.metric_values("loss", subset=TRAIN)

    model.load_state_dict(state)
    train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker_b
    )

    assert tracker_b.metric_values("loss", subset=TRAIN) == first


# -- realised outcomes ----------------------------------------------------
#
# The metrics a run is actually judged by. Whether the numbers are right is
# `test_policy.py`; what these assert is that the loop joins its predictions to
# the right queries and hands them over.


def outcomes_for(n=12):
    """A results table covering the queries `make_examples` builds.

    Odd queries are the ones only `bon` solved, so a run that learns anything at
    all can beat the always-baseline line - and one that predicts a single class
    cannot.
    """
    return outcome_table(
        *[
            query_rows(
                str(i),
                none=(i % 2 == 0, 100),
                bon=(True, 400),
                router=("bon", True, 420),
            )
            for i in range(n)
        ]
    )


def test_realised_outcomes_are_scored_when_an_outcome_table_is_given(
    model, encode, config
):
    tracker = RecordingTracker()

    result = train_router(
        make_examples(),
        model=model,
        encode=encode,
        config=config,
        tracker=tracker,
        outcomes=outcomes_for(),
    )

    assert len(result.evaluations) == 2 * 2  # two epochs, two scored subsets
    assert len(
        tracker.metric_values("realised_accuracy", subset=TRAIN, policy="model")
    ) == 2
    assert tracker.metric_values(
        "realised_accuracy", subset=VALIDATION, policy=ALWAYS_BASELINE
    ) == [0.5, 0.5]
    assert tracker.metric_values("coverage", subset=TRAIN, policy=ORACLE) == [1.0, 1.0]
    # A head this fresh predicts approaches the sweep never ran, and those
    # queries are unknown rather than wrong - which is exactly what coverage
    # below 1 is there to say.
    assert all(
        0.0 <= evaluation.score.coverage <= 1.0 for evaluation in result.evaluations
    )


def test_realised_accuracy_and_cost_are_the_runs_headline(model, encode, config):
    tracker = RecordingTracker()

    result = train_router(
        make_examples(),
        model=model,
        encode=encode,
        config=config,
        tracker=tracker,
        outcomes=outcomes_for(),
    )

    validation = [e for e in result.evaluations if e.subset == VALIDATION][-1]
    assert tracker.summaries["final/subset"] == VALIDATION
    assert tracker.summaries["final/realised_accuracy"] == validation.score.accuracy
    assert (
        tracker.summaries["final/cost_multiplier"] == validation.score.cost_multiplier
    )
    assert f"final/reference/{ORACLE}/realised_accuracy" in tracker.summaries


def test_without_an_outcome_table_a_run_still_trains_but_scores_nothing(
    model, encode, config
):
    tracker = RecordingTracker()

    result = train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker
    )

    assert result.evaluations == []
    assert tracker.metric_values("loss", subset=TRAIN)
    assert not [call for call in tracker.metrics if call.name == "realised_accuracy"]
    assert "final/realised_accuracy" not in tracker.summaries


def test_the_test_split_is_scored_only_when_a_run_asks_for_it(
    model, encode, config, tmp_path
):
    tracker = RecordingTracker()

    train_router(
        make_examples(),
        model=model,
        encode=encode,
        config=config,
        tracker=tracker,
        outcomes=outcomes_for(),
    )
    assert TEST not in {
        call.context.get("subset") for call in tracker.metrics
    }

    asked = RecordingTracker()
    train_router(
        make_examples(),
        model=model,
        encode=encode,
        config=replace(config, score_test=True),
        tracker=asked,
        outcomes=outcomes_for(),
    )

    assert asked.metric_values("realised_accuracy", subset=TEST, policy="model")
    assert asked.metric_values("realised_accuracy", subset=TEST, policy=ORACLE)
    # ... and it is reported as the test split, never folded into the headline.
    assert asked.summaries["final/subset"] == VALIDATION
    assert "test/realised_accuracy" in asked.summaries


def test_predictions_are_joined_to_queries_in_example_order_despite_shuffling():
    """A shuffled join would score a plausible-looking policy over wrong queries."""
    examples = [
        Example(
            query_id=str(i),
            dataset="gsm8k",
            text=chr(ord("a") + i),
            approach="none",
            label=0,
            split=TRAIN,
        )
        for i in range(8)
    ]

    def encode(texts):
        ids = torch.tensor([[ord(text[0]) % VOCAB] for text in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class ByFirstToken(torch.nn.Module):
        """Predicts a label read straight off the input, so order is checkable."""

        def forward(self, input_ids, attention_mask=None, effort=None):
            first = input_ids[:, 0] % len(APPROACHES)
            logits = torch.zeros(len(first), len(APPROACHES))
            logits[torch.arange(len(first)), first] = 10.0
            return logits

    loader = _loader(examples, encode, batch_size=3, shuffle=True)
    _, _, predictions = _run_epoch(
        ByFirstToken(),
        loader,
        torch.nn.CrossEntropyLoss(),
        None,
        device=torch.device("cpu"),
    )

    assert predictions == [
        (ord(e.text[0]) % VOCAB) % len(APPROACHES) for e in examples
    ]
