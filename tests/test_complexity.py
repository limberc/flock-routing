# tests/test_complexity.py
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from privacy_serving.complexity import compute_complexity, compute_complexity_detail
from privacy_serving.models import Message


def msgs(*contents: str) -> list[Message]:
    """Build a simple user-only message list."""
    return [Message(role="user", content=c) for c in contents]


@pytest.fixture(autouse=True)
def mock_perplexity():
    """
    Patch PerplexityScorer.raw_perplexity so tests don't load distilgpt2.

    Returns a realistic perplexity based on text length as a stand-in:
    short/simple text → low ppl (near 25), longer/denser text → higher ppl (up to 200).
    This keeps the combined score realistic enough to validate thresholds.
    """
    def fake_raw_perplexity(self, text: str) -> float:
        word_count = len(text.split())
        # Simple heuristic for test purposes: more words + certain keywords → higher ppl
        base = 25.0
        if any(kw in text.lower() for kw in ["theorem", "derive", "asymptotic", "O(n", "mergesort"]):
            base = 150.0
        elif "```" in text or "def " in text or "class " in text:
            base = 80.0
        elif word_count > 30:
            base = 60.0
        elif word_count > 10:
            base = 35.0
        return base

    with patch(
        "privacy_serving.complexity.perplexity.PerplexityScorer.raw_perplexity",
        fake_raw_perplexity,
    ):
        yield


# ── Threshold assertions ──────────────────────────────────────────────────────

def test_trivial_prompt_is_low():
    score = compute_complexity(msgs("Hi"))
    assert score < 0.20


def test_short_factual_is_low():
    score = compute_complexity(msgs("What is the capital of France?"))
    assert score < 0.45


def test_reasoning_prompt_is_high():
    score = compute_complexity(msgs(
        "Analyze the trade-offs between quicksort and mergesort. "
        "Prove that mergesort is O(n log n) in the worst case. "
        "Derive the recurrence and solve it step by step."
    ))
    assert score > 0.55


def test_code_content_raises_score():
    score = compute_complexity(msgs(
        "```python\ndef fibonacci(n):\n    pass\n```\n"
        "Implement fibonacci using dynamic programming and optimize it."
    ))
    assert score > 0.40


def test_math_content_raises_score():
    score = compute_complexity(msgs(
        "Given the asymptotic complexity O(n log n), derive the theorem and prove the bound."
    ))
    assert score > 0.40


# ── Relative ordering ─────────────────────────────────────────────────────────

def test_many_questions_score_higher_than_one():
    many = compute_complexity(msgs(
        "What is X? Why does Y happen? How does Z work? "
        "What are the implications? Can you explain the difference?"
    ))
    one = compute_complexity(msgs("What is X?"))
    assert many > one


# ── Bounds and edge cases ─────────────────────────────────────────────────────

def test_score_is_bounded():
    very_complex = msgs(
        "Analyze, compare, contrast, evaluate and critique. Prove and derive. "
        "Step by step, explain why and how. "
        "```python\nclass Foo:\n    def bar(self):\n        pass\n```\n" * 5
    )
    score = compute_complexity(very_complex)
    assert 0.0 <= score <= 1.0


def test_empty_messages_returns_zero():
    assert compute_complexity([]) == 0.0


def test_all_none_content_returns_zero():
    messages = [Message(role="user", content=None), Message(role="assistant", content=None)]
    assert compute_complexity(messages) == 0.0


# ── compute_complexity_detail ─────────────────────────────────────────────────

def test_detail_fields_are_present():
    detail = compute_complexity_detail(msgs("Hello world, how are you today?"))
    assert hasattr(detail, "final_score")
    assert hasattr(detail, "perplexity_score")
    assert hasattr(detail, "linguistic_score")
    assert hasattr(detail, "raw_perplexity")


def test_detail_final_score_matches_compute_complexity():
    messages = msgs("What is the capital of France?")
    assert compute_complexity(messages) == pytest.approx(
        compute_complexity_detail(messages).final_score
    )


def test_detail_empty_messages_returns_zero_detail():
    detail = compute_complexity_detail([])
    assert detail.final_score == 0.0
    assert detail.perplexity_score == 0.0
    assert detail.linguistic_score == 0.0
    assert detail.raw_perplexity == 0.0
