"""The finetuned router at serving time: a checkpoint, a device, and a prediction.

This is the serving half of the contract `training/optillm_router.py` documents.
The plugin in `plugins/` is a thin adapter over it - the part that knows about
optillm's calling convention - so everything worth testing lives here, where it
can be exercised with a tiny model and no server.

Three things are deliberate:

- **The architecture is the vendored one**, from `training/classifier.py`, the
  same definition the finetune serialised against. Serving a checkpoint through
  a re-derived class definition is how weights end up in mismatched layers with
  nothing raised.
- **The input text and the effort feature are the training ones**, built by
  `build_input_text` with `INFERENCE_EFFORT`. The model is never asked to predict
  under a condition it was not trained for.
- **The classifier is a cached singleton behind a lock.** Building it deserialises
  a ~400M-parameter encoder, and optillm serves concurrent requests from a thread
  pool: without the lock the first burst of requests would each load their own
  copy, and HuggingFace's fast tokenizer raises "Already borrowed" when called
  concurrently on one instance.

Both knobs are configuration, so swapping checkpoints or moving off the GPU needs
no code change and no container rebuild.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch

from router_lab.labels import FINETUNED_ROUTER
from router_lab.training.optillm_router import (
    APPROACHES,
    INFERENCE_EFFORT,
    build_input_text,
)

APPROACH = FINETUNED_ROUTER
"""The approach name this router is swept under - the plugin's slug."""

CHECKPOINT_ENV_VAR = "ROUTER_FT_CHECKPOINT"
DEVICE_ENV_VAR = "ROUTER_FT_DEVICE"


def resolve_device(requested: str | None = None) -> torch.device:
    """The device to serve on, from configuration, defaulting to a GPU.

    `auto` (the default) takes a GPU when one is present and falls back to CPU,
    which is what makes the same configuration work on an A100 node and on a 4GB
    laptop. Naming a device explicitly is a demand rather than a preference: a
    request for CUDA on a machine without it fails here, at load time, with the
    variable to change - not later, inside a request, as an invalid device
    ordinal from somewhere in torch.
    """
    requested = (requested or os.environ.get(DEVICE_ENV_VAR) or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"{DEVICE_ENV_VAR}={requested!r} is not a valid torch device: {exc}"
        ) from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{DEVICE_ENV_VAR}={requested!r} asks for a GPU, but torch reports "
                f"no CUDA device available. Set {DEVICE_ENV_VAR}=cpu, or 'auto' to "
                f"take whatever is present."
            )
        count = torch.cuda.device_count()
        if device.index is not None and device.index >= count:
            raise RuntimeError(
                f"{DEVICE_ENV_VAR}={requested!r} asks for GPU {device.index}, but "
                f"this machine has {count}."
            )
    return device


def checkpoint_path(configured: str | Path | None = None) -> Path:
    """Which checkpoint to serve. Configuration only - there is no default.

    A default would be worse than an error: it would serve some other run's
    router and record the results under this one's name.
    """
    path = configured or os.environ.get(CHECKPOINT_ENV_VAR)
    if not path:
        raise RuntimeError(
            f"no finetuned router checkpoint configured: set "
            f"${CHECKPOINT_ENV_VAR} to one written by train_router.py"
        )
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"${CHECKPOINT_ENV_VAR}={resolved} does not exist. Checkpoints are "
            f"written to checkpoints/router-<run>.pt; on the cluster the path "
            f"must be one a worker node can see."
        )
    return resolved


@dataclass
class FinetunedRouter:
    """A loaded classifier that answers "which approach for this query?".

    Holds the encoder seam rather than a tokenizer, exactly as the training loop
    does, so a test can drive it with a trivial encoder and no downloaded weights.
    """

    model: torch.nn.Module
    encode: object
    """An `Encoder` - texts in, `input_ids` and `attention_mask` out."""
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    lock: threading.Lock = field(default_factory=threading.Lock)
    """Serialises tokenise-and-predict; the LLM call afterwards still fans out."""
    checkpoint: Path | None = None
    """Recorded so a server can say which checkpoint it is actually serving."""

    def predict(self, system_prompt: str, query: str) -> str:
        """The approach to route this query to, by name.

        The text and the effort feature are the ones the finetune trained on, and
        come from the same functions that built its examples.
        """
        text = build_input_text(system_prompt, query)
        with self.lock:
            encoded = self.encode([text])
            self.model.eval()
            with torch.no_grad():
                logits = self.model(
                    encoded["input_ids"].to(self.device),
                    attention_mask=encoded["attention_mask"].to(self.device),
                    effort=torch.tensor(
                        [INFERENCE_EFFORT], dtype=torch.float, device=self.device
                    ),
                )
            index = int(logits.argmax(dim=1).item())
        return APPROACHES[index]


_cached: FinetunedRouter | None = None
_cache_lock = threading.Lock()


def load_router(
    checkpoint: str | Path | None = None, device: str | None = None
) -> FinetunedRouter:
    """The shared router, loaded once per process.

    Double-checked locking: the fast path is lock-free, and only the first
    (possibly concurrent) load is serialised. Loading is minutes of work and over
    a gigabyte of weights, so a per-request load would not merely be slow - it
    would exhaust the GPU under concurrency.
    """
    global _cached
    if _cached is None:
        with _cache_lock:
            if _cached is None:
                _cached = _load(checkpoint, device)
    return _cached


def _load(
    checkpoint: str | Path | None, device: str | None
) -> FinetunedRouter:
    from router_lab.training.classifier import (
        load_finetuned_router,
        tokenizer_encoder,
    )

    path = checkpoint_path(checkpoint)
    resolved = resolve_device(device)
    print(f"Loading finetuned router {path} on {resolved}")
    model, tokenizer = load_finetuned_router(path)
    model.to(resolved)
    return FinetunedRouter(
        model=model,
        encode=tokenizer_encoder(tokenizer),
        device=resolved,
        checkpoint=path,
    )


def reset_cache() -> None:
    """Drop the cached router. For tests, and for a server reloading a checkpoint."""
    global _cached
    with _cache_lock:
        _cached = None
