
import pytest

from src.evaluation.evaluate import normalize_text, load_questions


def test_normalize_text_converts_to_lowercase():
    result = normalize_text("Annual PAID Leave")
    assert result == {"annual", "paid", "leave"}


def test_normalize_text_removes_punctuation():
    result = normalize_text("Hello, world!")
    assert result == {"hello", "world"}


def test_normalize_text_removes_duplicate_words():
    result = normalize_text("leave leave policy")
    assert result == {"leave", "policy"}


def test_load_questions_returns_a_list():
    questions = load_questions()
    assert isinstance(questions, list)


def test_load_questions_contains_20_questions():
    questions = load_questions()
    assert len(questions) == 20
