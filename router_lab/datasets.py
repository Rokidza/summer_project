"""Benchmark datasets, declared as config rather than code.

Each entry says where the dataset lives, how to lay a row out into a prompt,
where the gold answer comes from, and which grading `category` its answers fit
(see `router_lab.grading`). Adding a benchmark whose answers fit an existing
category is one new entry here and nothing else.

Config keys:
    path, name, split   - the HuggingFace dataset coordinates.
    category            - selects system prompt, extraction, normalization,
                          and how correctness is decided.
    question_template   - str.format template over the row; builds the prompt.
    answer_key          - the row field holding the gold answer.
    gold_template       - optional; builds the gold from several row fields
                          instead of one (code datasets gold a test program).
    gold_split          - optional; keep only what follows this delimiter
                          (gsm8k's answer is a worked solution ending "#### 42").
    list_style          - optional; how list-valued fields render:
                          "lines" (newline-joined) or "lettered" ("A) ...").
                          A dict-valued field renders each of its values that
                          way, so a template can reach into it: "{choices[text]}".
    label_path          - optional; dotted path to the row field listing each
                          option's own label ("choices.label"). Lettered
                          rendering labels options by position, so where a
                          dataset carries its own labels the gold is the
                          position of `answer_key` in this list, not the
                          label's face value.
    id_key              - optional; the row field holding a stable query id.
                          Defaults to the row index.
"""

from __future__ import annotations

from dataclasses import dataclass

from router_lab.grading import normalize_answer

DATASETS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "category": "numeric",
        "question_template": "{question}",
        "answer_key": "answer",
        "gold_split": "####",
    },
    "math500": {
        "path": "HuggingFaceH4/MATH-500",
        "name": None,
        "split": "test",
        "category": "numeric",
        "question_template": "{problem}",
        "answer_key": "answer",
    },
    "aime24": {
        "path": "AI-MO/aimo-validation-aime",
        "name": None,
        "split": "train",
        "category": "numeric",
        "question_template": "{problem}",
        "answer_key": "answer",
    },
    "boolq": {
        "path": "google/boolq",
        "name": None,
        "split": "validation",
        "category": "boolean",
        "question_template": "Passage: {passage}\n\nQuestion: {question}",
        "answer_key": "answer",
    },
    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "test",
        "category": "multiple_choice",
        "question_template": "{question}\n\n{choices}",
        "list_style": {"choices": "lettered"},
        "answer_key": "answer",
    },
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "test",
        "category": "multiple_choice",
        "question_template": "{question}\n\n{choices[text]}",
        "list_style": {"choices": "lettered"},
        "answer_key": "answerKey",
        "label_path": "choices.label",
        "id_key": "id",
    },
    "humaneval": {
        "path": "openai/openai_humaneval",
        "name": None,
        "split": "test",
        "category": "code",
        "question_template": (
            "Complete the following Python function.\n\n```python\n{prompt}\n```"
        ),
        "answer_key": "test",
        # The gold is the dataset's own test program plus its invocation; the
        # grader runs the model's function against it.
        "gold_template": "{test}\n\ncheck({entry_point})",
        "id_key": "task_id",
    },
    "mbpp": {
        "path": "google-research-datasets/mbpp",
        "name": "full",
        "split": "test",
        "category": "code",
        # MBPP's asserts name the expected function, so the prompt must show
        # them or the model cannot know what to call its solution.
        "question_template": (
            "{text}\n\nYour solution must pass these tests:\n```python\n{test_list}\n```"
        ),
        "list_style": {"test_list": "lines"},
        "answer_key": "test_list",
        "gold_template": "{test_list}",
        "id_key": "task_id",
    },
}


@dataclass(frozen=True)
class Problem:
    """One benchmark query: what to ask, and what counts as right."""

    id: str
    question: str
    gold: str


def category_of(dataset_key: str) -> str:
    return DATASETS[dataset_key]["category"]


LETTERS = "ABCDEFGH"


def _render(value, style: str | None) -> str | dict:
    """Flatten a row field into template-ready text.

    A dict is kept as a dict of rendered values, so a template can select one
    of them by key ("{choices[text]}") rather than needing new config for every
    dataset that nests its options.
    """
    if isinstance(value, dict):
        return {key: _render(item, style) for key, item in value.items()}
    if not isinstance(value, (list, tuple)):
        return str(value)
    if style == "lettered":
        return "\n".join(
            f"{letter}) {item}" for letter, item in zip(LETTERS, value)
        )
    return "\n".join(str(item) for item in value)


def _style_of(cfg: dict, field: str) -> str | None:
    return cfg.get("list_style", {}).get(field)


def _fill(template: str, row: dict, cfg: dict) -> str:
    return template.format(
        **{key: _render(value, _style_of(cfg, key)) for key, value in row.items()}
    )


def _dig(row: dict, path: str):
    """Follow a dotted config path into a row: "choices.label"."""
    value = row
    for step in path.split("."):
        value = value[step]
    return value


def _gold_by_position(row: dict, cfg: dict) -> str:
    """The letter of the option the gold label names, by its position."""
    labels = [str(label) for label in _dig(row, cfg["label_path"])]
    gold_label = str(row[cfg["answer_key"]])
    if gold_label not in labels:
        raise ValueError(
            f"gold answer {gold_label!r} is not one of this row's option labels "
            f"{labels} (row id {row.get(cfg.get('id_key', ''), '?')})"
        )
    return LETTERS[labels.index(gold_label)]


def build_problem(index: int, row: dict, cfg: dict) -> Problem:
    """Turn one dataset row into a Problem, per its dataset's config."""
    if "gold_template" in cfg:
        gold = _fill(cfg["gold_template"], row, cfg)
    else:
        if "label_path" in cfg:
            gold = _gold_by_position(row, cfg)
        else:
            answer_key = cfg["answer_key"]
            gold = _render(row[answer_key], _style_of(cfg, answer_key))
            if "gold_split" in cfg:
                gold = gold.split(cfg["gold_split"])[-1]
        gold = normalize_answer(gold, cfg["category"])

    query_id = str(row[cfg["id_key"]]) if "id_key" in cfg else str(index)
    return Problem(id=query_id, question=_fill(cfg["question_template"], row, cfg), gold=gold)


def load_problems(dataset_key: str, limit: int | None = None) -> list[Problem]:
    """Load a dataset from HuggingFace and shape it into Problems."""
    from datasets import load_dataset

    cfg = DATASETS[dataset_key]
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"])
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [build_problem(i, row, cfg) for i, row in enumerate(ds)]
