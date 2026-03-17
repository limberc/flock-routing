# tests/test_complexity.py
import pytest
from privacy_serving.complexity import compute_complexity
from privacy_serving.models import Message


def msgs(*contents: str) -> list[Message]:
    """Build a simple user-only message list."""
    return [Message(role="user", content=c) for c in contents]


def test_trivial_prompt_is_low():
    score = compute_complexity(msgs("Hi"))
    assert score < 0.2


def test_short_factual_is_low():
    score = compute_complexity(msgs("What is the capital of France?"))
    assert score < 0.3


def test_reasoning_keywords_raise_score():
    score = compute_complexity(msgs(
        "Analyze the trade-offs between quicksort and mergesort. "
        "Prove that mergesort is O(n log n) in the worst case. "
        "Compare their space complexities step by step."
    ))
    assert score >= 0.4


def test_code_content_raises_score():
    score = compute_complexity(msgs(
        "```python\ndef fibonacci(n):\n    pass\n```\n"
        "Implement fibonacci using dynamic programming and optimize it."
    ))
    assert score >= 0.35


def test_math_content_raises_score():
    score = compute_complexity(msgs(
        "Given ∫f(x)dx = F(x) + C, derive the fundamental theorem of calculus."
    ))
    assert score >= 0.35


def test_many_questions_raise_score():
    score = compute_complexity(msgs(
        "What is X? Why does Y happen? How does Z work? "
        "What are the implications? Can you explain the difference?"
    ))
    assert score > compute_complexity(msgs("What is X?"))


def test_multi_turn_context_raises_score():
    many_messages = [Message(role="user", content=f"Message {i}") for i in range(12)]
    score_many = compute_complexity(many_messages)
    score_one = compute_complexity([Message(role="user", content="Message 0")])
    assert score_many > score_one


def test_score_is_bounded():
    very_complex = msgs(
        "Analyze, compare, contrast, evaluate and critique. Prove and derive. "
        "Step by step, explain why and how. ∑∫∂∇∈∀∃≤≥. "
        "```python\nclass Foo:\n    def bar(self):\n        pass\n```\n" * 10
    )
    score = compute_complexity(very_complex)
    assert 0.0 <= score <= 1.0


def test_empty_messages_returns_zero():
    score = compute_complexity([])
    assert score == 0.0
