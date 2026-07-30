"""The contract with optillm's published router checkpoint - the parts with no
deep-learning dependency. The architecture itself is in `classifier.py`.

Everything here is a copy of something in optillm's `router_plugin.py`, and that
duplication is the point. The checkpoint on the Hub was serialised against these
constants and that architecture; if the upstream checkout is refactored, weights
must not start loading into quietly mismatched layers, and training must not
start building input text the serving plugin will never send. So the definitions
live here, and `tests/test_optillm_router.py` fails when the upstream copy
drifts away from them.

Three of the four things that have to agree between training and serving are
here, and all three are silent when wrong rather than loud:

- **The input text.** `INPUT_TEMPLATE` is what the plugin concatenates. The
  system prompt is resolved through the dataset's category exactly as the
  harness resolves it when it runs a sweep - `tests/test_examples.py` pins that
  against what the harness actually puts on the wire.
- **The effort feature.** The plugin never passes one, so the model only ever
  sees `INFERENCE_EFFORT` in production; training under any other value trains
  for a condition that will not occur.
- **The label space.** `APPROACHES` in its original order, so index *i* means the
  same approach it meant to the pretrained head - which is what makes the head
  reusable instead of re-initialised.

Keeping this module free of torch is what lets example building - and
`train_router.py --dry-run` - work on an install without the training extras.
"""

from __future__ import annotations

APPROACHES = [
    "none",
    "mcts",
    "bon",
    "moa",
    "rto",
    "z3",
    "self_consistency",
    "pvg",
    "rstar",
    "cot_reflection",
    "plansearch",
    "leap",
    "re2",
]
"""The label space, in optillm's original order. Never sort or extend this."""

MAX_LENGTH = 1024
BASE_MODEL = "answerdotai/ModernBERT-large"
CHECKPOINT_REPO = "codelion/optillm-modernbert-large"

INFERENCE_EFFORT = 0.7
"""What the plugin's `predict_approach` defaults to, and is always called with."""

INPUT_TEMPLATE = "{system_prompt}\n\nUser: {query}"
"""The plugin's `preprocess_input` concatenation, verbatim."""

EFFORT_ENCODER_WIDTH = 64


def label_index(approach: str) -> int:
    """The head's output position for an approach name."""
    return APPROACHES.index(approach)


def build_input_text(system_prompt: str, query: str) -> str:
    """The exact string the serving plugin tokenises for this query."""
    return INPUT_TEMPLATE.format(system_prompt=system_prompt, query=query)
