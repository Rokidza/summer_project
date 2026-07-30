"""The training loop, exercised end to end without a network or a real router.

The encoder is a randomly-initialised ModernBERT built from a local config (see
`tests/tiny_router.py`), so nothing is downloaded; the tokenizer is replaced by
a trivial encoder at the seam the loop takes it through. Wiring is asserted -
convergence is not.
"""

import pytest

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
from router_lab.training.train import TrainingConfig, train_router  # noqa: E402
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


def test_records_per_epoch_loss_and_accuracy_under_a_subset_context(
    model, encode, config
):
    tracker = RecordingTracker()

    train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker
    )

    assert len(tracker.metric_values("loss", subset=TRAIN)) == 2
    assert len(tracker.metric_values("accuracy", subset=TRAIN)) == 2
    assert len(tracker.metric_values("loss", subset=VALIDATION)) == 2
    assert [call.step for call in tracker.metrics if call.name == "loss"] == [1, 1, 2, 2]
    assert all(0.0 <= value <= 1.0 for value in tracker.metric_values("accuracy", subset=TRAIN))


def test_the_test_split_is_never_scored(model, encode, config):
    tracker = RecordingTracker()

    train_router(
        make_examples(), model=model, encode=encode, config=config, tracker=tracker
    )

    assert {TEST} not in [set(c.values()) for c in tracker.contexts_of("accuracy")]


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
