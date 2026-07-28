"""Dataset config -> Problem, without touching the network.

Loading real datasets needs HuggingFace; building a problem out of a row does
not, and that is where the per-dataset config actually earns its keep.
"""

import pytest

from router_lab.datasets import DATASETS, build_problem, category_of


def test_a_single_field_question_is_used_verbatim():
    problem = build_problem(0, {"question": "2+2?", "answer": "reasoning #### 4"}, DATASETS["gsm8k"])

    assert problem.question == "2+2?"
    assert problem.gold == "4"  # split off the worked solution, then normalized


def test_a_multi_field_question_is_laid_out_by_its_template():
    problem = build_problem(
        0,
        {"passage": "Cats purr.", "question": "Do cats purr?", "answer": True},
        DATASETS["boolq"],
    )

    assert problem.question == "Passage: Cats purr.\n\nQuestion: Do cats purr?"
    assert problem.gold == "true"


def test_multiple_choice_options_are_rendered_as_lettered_lines():
    problem = build_problem(
        0,
        {"question": "Which is a mammal?", "choices": ["Cod", "Bat", "Ant", "Eel"], "answer": 1},
        DATASETS["mmlu"],
    )

    assert "A) Cod" in problem.question
    assert "B) Bat" in problem.question
    assert problem.gold == "B"


def test_a_code_dataset_golds_the_test_program_not_a_string_answer():
    problem = build_problem(
        0,
        {
            "prompt": "def add(a, b):\n",
            "test": "def check(f):\n    assert f(1, 2) == 3",
            "entry_point": "add",
            "task_id": "HumanEval/0",
        },
        DATASETS["humaneval"],
    )

    assert problem.id == "HumanEval/0"
    assert "def add(a, b):" in problem.question
    assert problem.gold.endswith("check(add)")


def test_list_valued_fields_render_as_lines():
    problem = build_problem(
        0,
        {
            "task_id": 601,
            "text": "Write add().",
            "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        },
        DATASETS["mbpp"],
    )

    assert problem.id == "601"
    assert "assert add(1, 2) == 3\nassert add(0, 0) == 0" in problem.question
    assert problem.gold == "assert add(1, 2) == 3\nassert add(0, 0) == 0"


def test_row_index_is_the_id_when_the_dataset_declares_no_id_field():
    problem = build_problem(7, {"problem": "x", "answer": "3"}, DATASETS["math500"])

    assert problem.id == "7"


@pytest.mark.parametrize("key", sorted(DATASETS))
def test_every_dataset_declares_a_category_with_grading_support(key):
    from router_lab.grading import EXTRACTORS, NORMALIZERS, SYSTEM_PROMPTS

    category = category_of(key)
    assert category in EXTRACTORS
    assert category in NORMALIZERS
    assert category in SYSTEM_PROMPTS


def test_the_suite_spans_more_than_maths():
    categories = {category_of(key) for key in DATASETS}

    assert {"numeric", "boolean", "code", "multiple_choice"} <= categories
