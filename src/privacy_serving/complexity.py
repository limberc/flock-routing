from __future__ import annotations

import tiktoken

from privacy_serving.models import Message

# Lazy-load tokenizer (cached after first use)
_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def _extract_text(messages: list[Message]) -> str:
    return " ".join(m.content or "" for m in messages)


# Keywords that strongly indicate reasoning / analytical tasks
_REASONING_KEYWORDS: list[str] = [
    "analyze", "analyse", "compare", "contrast", "evaluate", "critique",
    "step by step", "step-by-step", "prove", "derive", "explain why",
    "how does", "why does", "what causes", "implications", "implication",
    "trade-off", "tradeoff", "pros and cons", "advantages and disadvantages",
    "algorithm", "optimize", "optimise", "refactor", "implement",
    "design pattern", "architecture", "complexity", "theorem", "proof",
]

# Code indicators (backtick fences or common programming tokens)
_CODE_PATTERNS: list[str] = ["```", "def ", "class ", "import ", "function ", "const ", "var "]

# Math / formal notation indicators.
# Note: some words ("theorem", "proof", "derivative") also appear in
# _REASONING_KEYWORDS. This intentional overlap means deeply mathematical
# prompts score higher on both the reasoning (30%) and domain (25%) factors,
# which reflects their true difficulty.
_MATH_PATTERNS: list[str] = [
    "∑", "∫", "∂", "∇", "∈", "∀", "∃", "≤", "≥",
    "O(", "Θ(", "Ω(",
    "equation", "theorem", "proof", "formula", "matrix",
    "derivative", "integral", "calculus",
]


def compute_complexity(messages: list[Message]) -> float:
    """
    Return a prompt complexity score in [0.0, 1.0].

    Weights:
      - Token length        25%
      - Reasoning markers   30%
      - Code / math         25%
      - Question count      10%
      - Context depth       10%
    """
    if not messages:
        return 0.0

    text = _extract_text(messages)
    text_lower = text.lower()

    # 1. Length score: saturates at 2 000 tokens → score 1.0
    token_count = _count_tokens(text)
    length_score = min(token_count / 2000.0, 1.0)

    # 2. Reasoning markers: count distinct hits, saturate at 5
    reasoning_hits = sum(1 for kw in _REASONING_KEYWORDS if kw in text_lower)
    reasoning_score = min(reasoning_hits / 5.0, 1.0)

    # 3. Code + math detection
    # Any presence of code or math notation scores the full domain factor —
    # the factor measures whether structured formal notation is present, so
    # either signal alone is sufficient evidence.
    has_code = any(pat in text for pat in _CODE_PATTERNS)
    has_math = any(pat in text or pat in text_lower for pat in _MATH_PATTERNS)
    domain_score = 1.0 if (has_code or has_math) else 0.0

    # 4. Question multiplicity: saturates at 5 question marks
    question_count = text.count("?")
    question_score = min(question_count / 5.0, 1.0)

    # 5. Context depth: saturates at 10 messages
    depth_score = min(len(messages) / 10.0, 1.0)

    score = (
        length_score   * 0.25
        + reasoning_score * 0.30
        + domain_score    * 0.25
        + question_score  * 0.10
        + depth_score     * 0.10
    )
    return min(score, 1.0)
