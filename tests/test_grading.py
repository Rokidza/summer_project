"""Grading and answer extraction, per category.

These are pure text functions - no server, no store. Cases are drawn from the
ways models actually malform each answer shape.
"""

import pytest

from router_lab.grading import (
    extract_answer,
    is_correct,
    normalize_answer,
    strip_reasoning,
)


# -- reasoning blocks -----------------------------------------------------


def test_closed_reasoning_block_is_stripped():
    text = "<think>the answer might be 7</think>The answer is \\boxed{4}"

    assert strip_reasoning(text).strip() == "The answer is \\boxed{4}"


def test_unclosed_reasoning_block_keeps_only_what_follows_the_tag():
    """A truncated generation leaves <think> open; the scratchpad must not be graded."""
    assert strip_reasoning("prelude<think>maybe 7").strip() == "maybe 7"


# -- numeric --------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The answer is \\boxed{42}.", "42"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),  # brace-matched, not truncated
        ("\\boxed{12} then \\boxed{7}", "7"),  # the last box wins
        ("\\boxed{\\$1,200}", "1200"),
        ("\\boxed{12.0}", "12"),
        ("\\boxed{50\\%}", "50"),
        ("Steps... #### 18", "18"),
        ("no box, no hash, just 3 then 9", "9"),  # last number is the fallback
        ("<think>\\boxed{99}</think> \\boxed{5}", "5"),
        ("nothing numeric here", ""),
    ],
)
def test_numeric_answers_are_extracted_and_normalized(text, expected):
    assert extract_answer(text, "numeric") == expected


# -- boolean --------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("\\boxed{True}", "true"),
        ("\\boxed{no}", "false"),
        ("So the answer is False.", "false"),
        ("Yes, the passage supports it.", "true"),
        ("It is true. Actually, no.", "false"),  # the last mention wins
        ("unclear", ""),
    ],
)
def test_boolean_answers_are_extracted_and_normalized(text, expected):
    assert extract_answer(text, "boolean") == expected


# -- multiple choice ------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("\\boxed{B}", "B"),
        ("\\boxed{(C)}", "C"),
        ("The answer is B.", "B"),
        ("Answer: **D**", "D"),
        ("Option A is wrong, so the answer is C", "C"),
        ("I'd pick B", "B"),
        ("cannot determine", ""),
    ],
)
def test_choice_answers_are_extracted_and_normalized(text, expected):
    assert extract_answer(text, "multiple_choice") == expected


def test_choice_gold_answers_accept_letters_or_indices():
    assert normalize_answer("2", "multiple_choice") == "C"
    assert normalize_answer("(c)", "multiple_choice") == "C"


# -- code -----------------------------------------------------------------


def test_code_is_taken_from_the_last_fenced_block():
    text = (
        "Here's a first try:\n```python\ndef f(): return 0\n```\n"
        "Actually:\n```python\ndef f(): return 1\n```\nHope that helps!"
    )

    assert extract_answer(text, "code") == "def f(): return 1"


def test_unfenced_code_is_used_as_is():
    assert extract_answer("def f():\n    return 1", "code") == "def f():\n    return 1"


def test_code_is_graded_by_running_the_dataset_tests():
    solution = "def add(a, b):\n    return a + b"
    tests = "assert add(1, 2) == 3"

    assert is_correct(solution, tests, "code")


def test_code_failing_the_dataset_tests_is_incorrect():
    assert not is_correct("def add(a, b):\n    return a - b", "assert add(1, 2) == 3", "code")


def test_code_that_does_not_parse_is_incorrect_rather_than_raising():
    assert not is_correct("def add(a, b) return", "assert add(1, 2) == 3", "code")


def test_code_that_never_terminates_is_incorrect_rather_than_hanging():
    assert not is_correct(
        "def add(a, b):\n    while True: pass", "assert add(1, 2) == 3", "code", timeout=1.0
    )


def test_non_code_categories_are_graded_by_equality():
    assert is_correct("4", "4", "numeric")
    assert not is_correct("5", "4", "numeric")


def test_gold_answers_are_normalized_the_same_way_as_predictions():
    assert normalize_answer("True", "boolean") == extract_answer("\\boxed{yes}", "boolean")
    assert normalize_answer("1,200", "numeric") == extract_answer("\\boxed{1200}", "numeric")
