"""The finetuned router at serving time, with no server and no checkpoint.

The model is the tiny offline ModernBERT the training tests use, handed to
`FinetunedRouter` directly - so what is asserted here is the part a plugin cannot
be trusted to get right on its own: that serving builds the same input text the
finetune trained on, sends the same effort feature, and that the device and the
checkpoint really are configuration.
"""

import pytest

from router_lab.labels import FINETUNED_ROUTER

torch = pytest.importorskip("torch")

from router_lab.finetuned_router import (  # noqa: E402
    APPROACH,
    CHECKPOINT_ENV_VAR,
    DEVICE_ENV_VAR,
    FinetunedRouter,
    checkpoint_path,
    load_router,
    reset_cache,
    resolve_device,
)
from router_lab.training.optillm_router import (  # noqa: E402
    APPROACHES,
    INFERENCE_EFFORT,
    build_input_text,
)
from tests.tiny_router import tiny_classifier  # noqa: E402


@pytest.fixture(autouse=True)
def no_cached_router():
    """A cached singleton must not leak between tests."""
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def recorder():
    """An encoder that keeps what it was asked to tokenise."""

    class Recorder:
        def __init__(self):
            self.texts = []

        def __call__(self, texts):
            self.texts.extend(texts)
            ids = torch.ones(len(texts), 4, dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    return Recorder()


@pytest.fixture
def router(recorder):
    return FinetunedRouter(model=tiny_classifier(), encode=recorder)


def test_the_approach_name_is_the_slug_the_sweep_asks_for():
    assert APPROACH == FINETUNED_ROUTER == "router_ft"


def test_a_prediction_is_an_approach_in_optillms_label_space(router):
    assert router.predict("Be careful.", "2 + 2?") in APPROACHES


def test_serving_tokenises_the_text_the_finetune_trained_on(router, recorder):
    """The first of the four things training and serving must agree on."""
    router.predict("Be careful.", "2 + 2?")

    assert recorder.texts == [build_input_text("Be careful.", "2 + 2?")]


def test_the_effort_feature_is_the_constant_the_model_was_trained_under(recorder):
    seen = {}

    class RecordingModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, effort=None):
            seen["effort"] = effort
            return torch.zeros(len(input_ids), len(APPROACHES))

    FinetunedRouter(model=RecordingModel(), encode=recorder).predict("p", "q")

    # float32, so approx: what matters is that it is the training constant and
    # not some other default the plugin invented.
    assert seen["effort"].tolist() == pytest.approx([INFERENCE_EFFORT])


def test_the_highest_scoring_class_is_the_one_routed_to(recorder):
    class Decided(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, effort=None):
            logits = torch.zeros(len(input_ids), len(APPROACHES))
            logits[:, APPROACHES.index("moa")] = 10.0
            return logits

    router = FinetunedRouter(model=Decided(), encode=recorder)

    assert router.predict("p", "q") == "moa"


# -- configuration --------------------------------------------------------


def test_the_device_defaults_to_a_gpu_when_there_is_one(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert resolve_device().type == expected


def test_cpu_can_be_demanded_so_the_constrained_laptop_path_survives(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")

    assert resolve_device().type == "cpu"

    # An argument still wins over the environment: the plugin is configured by
    # whoever starts the server, but a caller may know better.
    assert resolve_device("cpu").type == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="a GPU is present")
def test_a_gpu_that_is_not_there_fails_by_name_not_by_ordinal(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda")

    with pytest.raises(RuntimeError, match=f"{DEVICE_ENV_VAR}='cuda' asks for a GPU"):
        resolve_device()


def test_a_device_that_is_not_a_device_says_so(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "tpu")

    with pytest.raises(ValueError, match="is not a valid torch device"):
        resolve_device()


def test_the_checkpoint_comes_from_configuration(monkeypatch, tmp_path):
    written = tmp_path / "router-test.pt"
    written.write_bytes(b"")
    monkeypatch.setenv(CHECKPOINT_ENV_VAR, str(written))

    assert checkpoint_path() == written
    # Changing checkpoints is changing a variable, not rebuilding anything.
    other = tmp_path / "router-other.pt"
    other.write_bytes(b"")
    assert checkpoint_path(other) == other


def test_an_unconfigured_checkpoint_is_refused_rather_than_defaulted(monkeypatch):
    """A default would serve some other run's router under this one's name."""
    monkeypatch.delenv(CHECKPOINT_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError, match=CHECKPOINT_ENV_VAR):
        checkpoint_path()


def test_a_checkpoint_that_is_not_there_fails_at_load_time(monkeypatch, tmp_path):
    monkeypatch.setenv(CHECKPOINT_ENV_VAR, str(tmp_path / "missing.pt"))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_router()


def test_the_router_is_loaded_once_and_reused_across_requests(monkeypatch, recorder):
    """Concurrent requests share one classifier - see the lock in `predict`."""
    loads = []

    def load(checkpoint, device):
        loads.append(checkpoint)
        return FinetunedRouter(model=tiny_classifier(), encode=recorder)

    monkeypatch.setattr("router_lab.finetuned_router._load", load)

    first, second = load_router(), load_router()

    assert first is second
    assert len(loads) == 1
