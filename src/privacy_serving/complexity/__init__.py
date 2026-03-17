# src/privacy_serving/complexity/__init__.py
# Temporary shim: reproduces heuristic scorer while new components are built.
# This file will be replaced entirely in Task 5.
from __future__ import annotations

import tiktoken

from privacy_serving.models import Message

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


_REASONING_KEYWORDS = [
    "analyze", "analyse", "compare", "contrast", "evaluate", "critique",
    "step by step", "step-by-step", "prove", "derive", "explain why",
    "how does", "why does", "what causes", "implications", "implication",
    "trade-off", "tradeoff", "pros and cons", "advantages and disadvantages",
    "algorithm", "optimize", "optimise", "refactor", "implement",
    "design pattern", "architecture", "complexity", "theorem", "proof",
]

_CODE_PATTERNS = ["```", "def ", "class ", "import ", "function ", "const ", "var "]

_MATH_PATTERNS = [
    "∑", "∫", "∂", "∇", "∈", "∀", "∃", "≤", "≥",
    "O(", "Θ(", "Ω(",
    "equation", "theorem", "proof", "formula", "matrix",
    "derivative", "integral", "calculus",
]


def compute_complexity(messages: list[Message]) -> float:
    if not messages:
        return 0.0
    text = " ".join(m.content or "" for m in messages)
    text_lower = text.lower()
    token_count = _count_tokens(text)
    length_score = min(token_count / 2000.0, 1.0)
    reasoning_hits = sum(1 for kw in _REASONING_KEYWORDS if kw in text_lower)
    reasoning_score = min(reasoning_hits / 5.0, 1.0)
    has_code = any(pat in text for pat in _CODE_PATTERNS)
    has_math = any(pat in text or pat in text_lower for pat in _MATH_PATTERNS)
    domain_score = 1.0 if (has_code or has_math) else 0.0
    question_score = min(text.count("?") / 5.0, 1.0)
    depth_score = min(len(messages) / 10.0, 1.0)
    return min(
        length_score * 0.25
        + reasoning_score * 0.30
        + domain_score * 0.25
        + question_score * 0.10
        + depth_score * 0.10,
        1.0,
    )
