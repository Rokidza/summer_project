"""Answer extraction and grading, one implementation per answer *category*.

A dataset declares a category (see `router_lab.datasets`); the category selects
the system prompt, the extractor that pulls a prediction out of the model's
prose, and the normalizer that puts prediction and gold answer into the same
shape so they can be compared with `==`.

Adding a dataset whose answers fit an existing category needs no code here -
only a new entry in DATASETS. A genuinely new answer shape adds one entry to
each of the three tables below, and becomes reusable by the next dataset of
that shape.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
BOOLEAN_RE = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)
CHOICE_RE = re.compile(r"\b([A-D])\b")
FENCE_RE = re.compile(r"```(?:[Pp]ython|py)?\s*\n(.*?)```", re.DOTALL)


SYSTEM_PROMPTS = {
    "numeric": (
        "You are a careful mathematical problem solver. Reason step by step, "
        "then give your final answer inside \\boxed{}."
    ),
    "boolean": (
        "You are a careful reading-comprehension assistant. Read the passage, "
        "reason step by step, then answer the question with exactly one word: "
        "True or False."
    ),
    "multiple_choice": (
        "You are a careful expert answering a multiple-choice question. Reason "
        "step by step, then give the letter of the single best option inside "
        "\\boxed{}."
    ),
    "code": (
        "You are an expert Python programmer. Complete the requested function. "
        "Return the full function definition in a single ```python code block, "
        "with no explanation after it."
    ),
}


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks so we grade the answer, not the scratchpad."""
    text = THINK_RE.sub("", text)
    # Unclosed <think> (truncated generation): keep only what follows the tag.
    if "<think>" in text.lower():
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[-1]
    return text


def extract_boxed(text: str) -> str | None:
    """Find the last \\boxed{...}, brace-matched so \\frac{1}{2} survives."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    start = text.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


# -- normalizers ----------------------------------------------------------


def normalize_numeric(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    # Escaped forms first: stripping "$" before "\$" would strand the backslash.
    ans = ans.replace("\\$", "").replace("\\%", "")
    ans = ans.replace("$", "").replace("%", "").replace(",", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.strip().rstrip(".").strip()
    ans = re.sub(r"^\\text\{(.*)\}$", r"\1", ans).strip()
    # 12.0 -> 12 so integer comparisons don't fail on formatting
    if re.fullmatch(r"-?\d+\.0+", ans):
        ans = ans.split(".")[0]
    return ans


def normalize_boolean(ans: str) -> str:
    ans = ans.strip().lower()
    if ans in ("true", "yes", "1"):
        return "true"
    if ans in ("false", "no", "0"):
        return "false"
    return ans


def normalize_choice(ans: str) -> str:
    """Letter answers compare case-insensitively, with any wrapping stripped.

    Datasets store the key as a letter ("B") or as a 0-based index ("1"); both
    normalize to the same upper-case letter.
    """
    ans = ans.strip().strip(".()[] ").upper()
    if re.fullmatch(r"[0-3]", ans):
        return "ABCD"[int(ans)]
    match = CHOICE_RE.search(ans)
    return match.group(1) if match else ans


def normalize_code(ans: str) -> str:
    """Code is graded by execution, so normalization only trims whitespace."""
    return ans.strip()


NORMALIZERS = {
    "numeric": normalize_numeric,
    "boolean": normalize_boolean,
    "multiple_choice": normalize_choice,
    "code": normalize_code,
}


# -- extractors -----------------------------------------------------------


def extract_numeric_answer(text: str) -> str:
    text = strip_reasoning(text)
    boxed = extract_boxed(text)
    if boxed is not None:
        return normalize_numeric(boxed)
    if "####" in text:
        return normalize_numeric(text.split("####")[-1].split("\n")[0])
    numbers = NUMBER_RE.findall(text.replace(",", ""))
    return normalize_numeric(numbers[-1]) if numbers else ""


def extract_boolean_answer(text: str) -> str:
    text = strip_reasoning(text)
    boxed = extract_boxed(text)
    if boxed is not None:
        match = BOOLEAN_RE.search(boxed)
        if match:
            return normalize_boolean(match.group(1))
    matches = BOOLEAN_RE.findall(text)
    return normalize_boolean(matches[-1]) if matches else ""


def extract_choice_answer(text: str) -> str:
    """Pull an option letter out, preferring \\boxed{} then a trailing letter.

    Models routinely answer "B", "(B)", "**B**", "Answer: B", or bury the letter
    at the end of a sentence; all of those land on the same letter.
    """
    text = strip_reasoning(text)
    boxed = extract_boxed(text)
    if boxed is not None:
        letter = normalize_choice(boxed)
        if re.fullmatch(r"[A-D]", letter):
            return letter
    labelled = re.findall(
        r"(?:answer|option)\s*(?:is)?\s*[:\-]?\s*\(?\*{0,2}([A-D])\b",
        text,
        re.IGNORECASE,
    )
    if labelled:
        return labelled[-1].upper()
    matches = CHOICE_RE.findall(text)
    return matches[-1].upper() if matches else ""


def extract_code_answer(text: str) -> str:
    """Pull the model's program out: the last fenced block, else the raw text.

    Prose around the code is discarded, but the code itself is left byte-exact -
    grading runs it, so nothing may be reformatted away.
    """
    text = strip_reasoning(text)
    blocks = FENCE_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


EXTRACTORS = {
    "numeric": extract_numeric_answer,
    "boolean": extract_boolean_answer,
    "multiple_choice": extract_choice_answer,
    "code": extract_code_answer,
}


def extract_answer(text: str, category: str) -> str:
    """The model's prediction, in the shape the category's gold answers use."""
    return EXTRACTORS[category](text)


def normalize_answer(answer: str, category: str) -> str:
    """A gold answer, in the same shape extraction produces."""
    return NORMALIZERS[category](answer)


def system_prompt(category: str) -> str:
    return SYSTEM_PROMPTS[category]


# -- correctness ----------------------------------------------------------

CODE_TIMEOUT_S = 10.0


def run_code_tests(program: str, tests: str, timeout: float = CODE_TIMEOUT_S) -> bool:
    """Run the dataset's asserts against the model's program; True if they pass.

    The model's code is executed in a separate interpreter with a wall-clock
    timeout, so a syntax error, an exception or an infinite loop all read as a
    failed solution rather than taking the sweep down with them.

    This *does* execute model-generated code on this machine. That is inherent
    to grading a code benchmark; the subprocess is an isolation and termination
    boundary, not a security sandbox. Only run trusted benchmark suites.
    """
    with tempfile.TemporaryDirectory() as workdir:
        script = Path(workdir) / "candidate.py"
        script.write_text(f"{program}\n\n{tests}\n")
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=workdir,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0


def is_correct(
    predicted: str, gold: str, category: str, timeout: float = CODE_TIMEOUT_S
) -> bool:
    """Did the model answer this query correctly?

    Most categories compare normalized strings. Code is different in kind: the
    gold "answer" is the dataset's test program, and correctness means the
    model's program passes it.
    """
    if category == "code":
        if not predicted.strip():
            return False
        return run_code_tests(predicted, gold, timeout)
    return predicted == gold
