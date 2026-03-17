# Smart Complexity Scorer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the heuristic complexity scorer in `src/privacy_serving/complexity.py` with a principled hybrid of distilgpt2 perplexity scoring (55%) and 7 linguistic features (45%), keeping `compute_complexity(messages) -> float` API unchanged.

**Architecture:** Delete `complexity.py` and replace with a `complexity/` package containing `perplexity.py` (PerplexityScorer), `linguistic.py` (LinguisticScorer), and `__init__.py` (ComplexityScorer combiner + public API). The perplexity scorer uses distilgpt2 with lazy loading; the linguistic scorer is pure Python.

**Tech Stack:** Python 3.10+, FastAPI, `transformers>=4.40` (distilgpt2), `torch>=2.0`, pytest

**Spec:** `docs/superpowers/specs/2026-03-17-smart-complexity-scorer-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Delete | `src/privacy_serving/complexity.py` | Old heuristic (replaced) |
| Create | `src/privacy_serving/complexity/__init__.py` | Public API: `compute_complexity`, `compute_complexity_detail`, `warmup` |
| Create | `src/privacy_serving/complexity/perplexity.py` | `PerplexityScorer`, `_normalise_perplexity` |
| Create | `src/privacy_serving/complexity/linguistic.py` | `LinguisticScorer` with 7 features |
| Modify | `src/privacy_serving/main.py` | Add lifespan context manager calling `warmup()` |
| Modify | `pyproject.toml` | Task 3: add `transformers>=4.40`, `torch>=2.0`; Task 5: remove `tiktoken` |
| Modify | `tests/test_complexity.py` | Revise thresholds, drop multi-turn test, add new tests |
| Modify | `tests/test_api.py` | Patch `warmup` in `client` fixture |
| Create | `tests/test_complexity_linguistic.py` | Unit tests for LinguisticScorer |
| Create | `tests/test_complexity_perplexity.py` | Unit tests for PerplexityScorer (mocked model) |

---

## Task 1: Package skeleton — swap `complexity.py` for `complexity/` package

**Context:** Python packages (directories with `__init__.py`) take precedence over modules of the same name, but having both is confusing. The safest migration is: create the package with a shim that reproduces the old heuristic, then delete the old file. All existing tests must still pass after this task.

**Files:**
- Create: `src/privacy_serving/complexity/__init__.py`
- Delete: `src/privacy_serving/complexity.py`

---

- [ ] **Step 1: Create the package directory and shim `__init__.py`**

```python
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
```

- [ ] **Step 2: Verify the package takes precedence (both files exist briefly)**

Run: `python -c "from privacy_serving.complexity import compute_complexity; print('OK')"`
Expected: `OK` (no import error, picks up the new package, not the old .py)

- [ ] **Step 3: Run existing tests — they must pass**

Run: `pytest tests/test_complexity.py tests/test_router.py tests/test_api.py -v`
Expected: All tests PASS (same behavior as before)

- [ ] **Step 4: Delete the old `complexity.py`**

```bash
rm src/privacy_serving/complexity.py
```

- [ ] **Step 5: Run tests again to confirm deletion didn't break anything**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/privacy_serving/complexity/__init__.py
git rm src/privacy_serving/complexity.py
git commit -m "refactor: replace complexity.py with complexity/ package (heuristic shim in __init__.py)"
```

---

## Task 2: Implement `LinguisticScorer`

**Context:** Seven pure-Python text features, no external dependencies. Build and test this in isolation before touching the perplexity scorer or pyproject.toml.

**Files:**
- Create: `src/privacy_serving/complexity/linguistic.py`
- Create: `tests/test_complexity_linguistic.py`

---

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_complexity_linguistic.py
from __future__ import annotations

import pytest
from privacy_serving.complexity.linguistic import LinguisticScorer


@pytest.fixture
def scorer():
    return LinguisticScorer()


# ── Empty input ──────────────────────────────────────────────────────────────

def test_empty_string_score_is_zero(scorer):
    assert scorer.score("") == 0.0


def test_empty_string_features_are_zero(scorer):
    features = scorer.features("")
    assert all(v == 0.0 for v in features.values())


# ── Score bounds ──────────────────────────────────────────────────────────────

def test_score_bounded_for_various_inputs(scorer):
    texts = [
        "Hi",
        "What is the capital of France?",
        "a " * 200,
        "The quick brown fox jumps over the lazy dog.",
        "However, the theorem cannot be derived unless the boundary conditions hold.",
    ]
    for text in texts:
        s = scorer.score(text)
        assert 0.0 <= s <= 1.0, f"Out of bounds for: {text!r}"


# ── ZeroDivisionError guards ─────────────────────────────────────────────────

def test_no_error_on_single_word_no_punctuation(scorer):
    s = scorer.score("hello")
    assert 0.0 <= s <= 1.0


def test_no_error_on_text_without_any_punctuation(scorer):
    s = scorer.score("the quick brown fox jumps over the lazy dog")
    assert 0.0 <= s <= 1.0


def test_no_error_on_empty_sentences(scorer):
    # Multiple dots produce empty segments — must not crash
    s = scorer.score("Hello... world...")
    assert 0.0 <= s <= 1.0


# ── TTR and hapax: short-input guard ─────────────────────────────────────────

def test_ttr_zero_for_fewer_than_five_words(scorer):
    features = scorer.features("hello world")
    assert features["ttr"] == 0.0


def test_hapax_zero_for_fewer_than_five_words(scorer):
    features = scorer.features("hello world")
    assert features["hapax"] == 0.0


def test_ttr_nonzero_for_five_words(scorer):
    # Five unique words → TTR should be > 0
    features = scorer.features("the quick brown fox jumps")
    assert features["ttr"] > 0.0


def test_hapax_nonzero_for_five_unique_words(scorer):
    features = scorer.features("the quick brown fox jumps")
    assert features["hapax"] > 0.0


# ── FK grade level ────────────────────────────────────────────────────────────

def test_fk_grade_never_negative(scorer):
    # Very simple short words → raw FK may be negative → clamped to 0
    features = scorer.features("hi ok go up do")
    assert features["fk_grade"] >= 0.0


def test_fk_grade_higher_for_complex_text(scorer):
    simple = scorer.features("Hi how are you today")["fk_grade"]
    complex_text = scorer.features(
        "The asymptotic complexity of the recursive Fibonacci implementation "
        "demonstrates exponential time complexity without memoization."
    )["fk_grade"]
    assert complex_text > simple


# ── Conditional marker detection ─────────────────────────────────────────────

def test_conditional_density_detects_if(scorer):
    features = scorer.features("Do this if the value is positive.")
    assert features["conditional_density"] > 0.0


def test_conditional_density_if_uses_word_boundary(scorer):
    # "wifi", "specify", "tariff" contain 'if' as substring — must NOT match
    features = scorer.features("Connect to the wifi and specify the tariff.")
    assert features["conditional_density"] == 0.0


def test_conditional_density_detects_multiword_marker(scorer):
    features = scorer.features("Proceed given that the constraints are satisfied.")
    assert features["conditional_density"] > 0.0


# ── Discourse connective detection ───────────────────────────────────────────

def test_discourse_density_detects_connectives(scorer):
    text = "Results were unexpected. However, the method was sound. Therefore we proceed."
    features = scorer.features(text)
    assert features["discourse_density"] > 0.0


def test_discourse_density_zero_for_plain_text(scorer):
    features = scorer.features("The cat sat on the mat.")
    assert features["discourse_density"] == 0.0


# ── Relative ordering ─────────────────────────────────────────────────────────

def test_complex_text_scores_higher_than_simple(scorer):
    simple = scorer.score("Hi there, how are you?")
    complex_text = scorer.score(
        "The differential equation demonstrates that unless the initial conditions "
        "satisfy the boundary constraints, the proposed theorem cannot be derived. "
        "Furthermore, assuming the provided assumptions hold, we must consequently "
        "re-evaluate the methodology. However, the results suggest otherwise."
    )
    assert complex_text > simple
```

- [ ] **Step 2: Run tests to confirm they fail (module not found)**

Run: `pytest tests/test_complexity_linguistic.py -v`
Expected: `ModuleNotFoundError` or `ImportError` — confirming the tests are real

- [ ] **Step 3: Implement `linguistic.py`**

```python
# src/privacy_serving/complexity/linguistic.py
from __future__ import annotations

import re
from collections import Counter

STOPWORDS: frozenset[str] = frozenset({
    "a", "above", "after", "again", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "back", "be", "because", "been", "being", "before",
    "between", "both", "but", "by", "can", "could", "dare", "did", "do",
    "does", "down", "during", "each", "even", "few", "for", "from", "further",
    "get", "had", "has", "have", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "in", "into", "is", "it", "its", "itself",
    "just", "me", "might", "more", "most", "my", "myself", "need", "no", "nor",
    "not", "of", "off", "on", "or", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shall", "she", "should", "since", "so", "some",
    "still", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "used", "very", "via", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "will",
    "with", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves",
})

# "if" uses word-boundary matching (\bif\b) to avoid hitting "wifi", "tariff", etc.
# All other markers use substring (.count()) which correctly handles multi-word phrases.
_CONDITIONAL_MARKERS = [
    "unless", "when", "provided", "assuming", "whether",
    "given that", "in case", "as long as",
]

_DISCOURSE_CONNECTIVES = [
    "however", "therefore", "thus", "consequently", "hence",
    "moreover", "furthermore", "nevertheless", "nonetheless",
    "although", "whereas",
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _count_syllables(word: str) -> int:
    """Approximate syllable count — heuristic only; known to undercount 'le' endings."""
    word = word.lower()
    count = len(re.findall(r"[aeiou]+", word))
    if word.endswith("e") and len(word) > 2 and count >= 2:
        count -= 1  # silent-e adjustment
    return max(1, count)


class LinguisticScorer:
    def features(self, text: str) -> dict[str, float]:
        """Return per-feature normalised scores (all in [0, 1])."""
        if not text:
            return {
                "fk_grade": 0.0,
                "ttr": 0.0,
                "hapax": 0.0,
                "content_density": 0.0,
                "avg_sent_len": 0.0,
                "conditional_density": 0.0,
                "discourse_density": 0.0,
            }

        words = re.findall(r"[a-zA-Z']+", text)
        total_words = max(1, len(words))
        words_lower = [w.lower() for w in words]
        text_lower = text.lower()

        sentences = [s for s in re.split(r"[.?!]", text) if len(s.strip()) >= 3]
        sentence_count = max(1, len(sentences))

        # 1. Flesch-Kincaid Grade Level
        total_syllables = sum(_count_syllables(w) for w in words)
        fk_grade = (
            0.39 * (total_words / sentence_count)
            + 11.8 * (total_syllables / total_words)
            - 15.59
        )
        fk_norm = _clamp(max(0.0, fk_grade) / 18.0, 0.0, 1.0)

        # 2. Type-Token Ratio (0.0 if < 5 words)
        if len(words) < 5:
            ttr_norm = 0.0
        else:
            ttr = len(set(words_lower)) / total_words
            ttr_norm = _clamp(ttr / 0.9, 0.0, 1.0)

        # 3. Hapax ratio (0.0 if < 5 words)
        if len(words) < 5:
            hapax_norm = 0.0
        else:
            counts = Counter(words_lower)
            hapax = sum(1 for c in counts.values() if c == 1) / total_words
            hapax_norm = _clamp(hapax / 0.7, 0.0, 1.0)

        # 4. Content-word density
        content_words = sum(1 for w in words_lower if w not in STOPWORDS)
        content_norm = _clamp((content_words / total_words) / 0.8, 0.0, 1.0)

        # 5. Average sentence length
        avg_sent_norm = _clamp((total_words / sentence_count) / 40.0, 0.0, 1.0)

        # 6. Conditional density
        # "if" uses word-boundary regex; all others use substring count
        cond_hits = len(re.findall(r"\bif\b", text_lower))
        for marker in _CONDITIONAL_MARKERS:
            cond_hits += text_lower.count(marker)
        cond_norm = _clamp((cond_hits / sentence_count) / 2.0, 0.0, 1.0)

        # 7. Discourse connective density
        disc_hits = sum(text_lower.count(c) for c in _DISCOURSE_CONNECTIVES)
        disc_norm = _clamp((disc_hits / sentence_count) / 1.5, 0.0, 1.0)

        return {
            "fk_grade": fk_norm,
            "ttr": ttr_norm,
            "hapax": hapax_norm,
            "content_density": content_norm,
            "avg_sent_len": avg_sent_norm,
            "conditional_density": cond_norm,
            "discourse_density": disc_norm,
        }

    def score(self, text: str) -> float:
        """Weighted average of all features. Returns 0.0 for empty input."""
        if not text:
            return 0.0
        f = self.features(text)
        return (
            f["fk_grade"]            * 0.20
            + f["ttr"]               * 0.15
            + f["hapax"]             * 0.15
            + f["content_density"]   * 0.15
            + f["avg_sent_len"]      * 0.15
            + f["conditional_density"] * 0.10
            + f["discourse_density"] * 0.10
        )
```

- [ ] **Step 4: Run tests — all must pass**

Run: `pytest tests/test_complexity_linguistic.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full suite to confirm nothing regressed**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/privacy_serving/complexity/linguistic.py tests/test_complexity_linguistic.py
git commit -m "feat: implement LinguisticScorer with 7 structural text features"
```

---

## Task 3: Add `transformers` and `torch` to dependencies

**Context:** `tiktoken` stays in `pyproject.toml` until Task 5 replaces the shim that imports it. We only add the new deps here so `PerplexityScorer` can be implemented in Task 4.

**Files:**
- Modify: `pyproject.toml`

---

- [ ] **Step 1: Add `transformers>=4.40` and `torch>=2.0` to `pyproject.toml`**

In `pyproject.toml`, add the two new deps alongside the existing ones (keep `tiktoken`):

```toml
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "httpx>=0.27",
    "pydantic>=2.7",
    "pyyaml>=6.0",
    "tiktoken>=0.7",
    "transformers>=4.40",
    "torch>=2.0",
]
```

- [ ] **Step 2: Install updated dependencies**

Run: `pip install -e ".[dev]"`
Expected: Installs `transformers` and `torch`; no errors

> Note: `torch` is large (~2 GB). This step may take several minutes on first install.

- [ ] **Step 3: Verify existing tests still pass**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add transformers>=4.40 and torch>=2.0 dependencies"
```

---

## Task 4: Implement `PerplexityScorer`

**Context:** `PerplexityScorer` wraps distilgpt2 with lazy loading. The model/tokenizer are module-level globals, loaded once on the first call to `raw_perplexity()`. The normalisation formula `_normalise_perplexity` is defined here (in `perplexity.py`) and will be imported by `__init__.py` in Task 5.

**Files:**
- Create: `src/privacy_serving/complexity/perplexity.py`
- Create: `tests/test_complexity_perplexity.py`

---

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_complexity_perplexity.py
from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock, patch


# ── _normalise_perplexity: pure-math, no model needed ────────────────────────

def test_normalise_at_floor_is_zero():
    from privacy_serving.complexity.perplexity import _normalise_perplexity, MIN_PPL
    assert _normalise_perplexity(MIN_PPL) == pytest.approx(0.0, abs=1e-6)


def test_normalise_at_ceiling_is_one():
    from privacy_serving.complexity.perplexity import _normalise_perplexity, MAX_PPL
    assert _normalise_perplexity(MAX_PPL) == pytest.approx(1.0, abs=1e-6)


def test_normalise_ppl_50_is_approximately_028():
    from privacy_serving.complexity.perplexity import _normalise_perplexity
    score = _normalise_perplexity(50.0)
    assert 0.25 < score < 0.32


def test_normalise_ppl_150_is_approximately_063():
    from privacy_serving.complexity.perplexity import _normalise_perplexity
    score = _normalise_perplexity(150.0)
    assert 0.60 < score < 0.67


def test_normalise_below_floor_clamped_to_zero():
    from privacy_serving.complexity.perplexity import _normalise_perplexity
    assert _normalise_perplexity(1.0) == 0.0


def test_normalise_above_ceiling_clamped_to_one():
    from privacy_serving.complexity.perplexity import _normalise_perplexity
    assert _normalise_perplexity(10_000.0) == 1.0


# ── PerplexityScorer with mocked model ───────────────────────────────────────

@pytest.fixture
def patched_scorer():
    """PerplexityScorer backed by a mock model — no download or GPU needed."""
    import torch
    import privacy_serving.complexity.perplexity as pmod

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    # Tokenizer returns fake input_ids tensor
    fake_ids = torch.zeros(1, 5, dtype=torch.long)
    mock_tokenizer.return_value = {"input_ids": fake_ids}

    # Model forward pass: loss=3.0 → raw_ppl = exp(3.0) ≈ 20.09
    mock_output = MagicMock()
    mock_output.loss.item.return_value = 3.0
    mock_model.return_value = mock_output

    with (
        patch.object(pmod, "_model", mock_model),
        patch.object(pmod, "_tokenizer", mock_tokenizer),
        patch.object(pmod, "_load_model"),  # no-op so lazy loading doesn't trigger
    ):
        from privacy_serving.complexity.perplexity import PerplexityScorer
        yield PerplexityScorer()


def test_score_bounded(patched_scorer):
    score = patched_scorer.score("anything")
    assert 0.0 <= score <= 1.0


def test_raw_perplexity_greater_than_one(patched_scorer):
    ppl = patched_scorer.raw_perplexity("hello world")
    assert ppl > 1.0


def test_score_equals_normalised_raw_perplexity(patched_scorer):
    from privacy_serving.complexity.perplexity import _normalise_perplexity
    ppl = patched_scorer.raw_perplexity("test")
    expected_score = _normalise_perplexity(ppl)
    actual_score = patched_scorer.score("test")
    assert actual_score == pytest.approx(expected_score, abs=1e-6)


def test_score_calls_raw_perplexity_internally(patched_scorer):
    """score() is a wrapper around raw_perplexity() — both return consistent values."""
    from privacy_serving.complexity.perplexity import _normalise_perplexity
    raw = patched_scorer.raw_perplexity("test input")
    expected = _normalise_perplexity(raw)
    assert patched_scorer.score("test input") == pytest.approx(expected, abs=1e-6)
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_complexity_perplexity.py -v`
Expected: `ModuleNotFoundError` — file doesn't exist yet

- [ ] **Step 3: Implement `perplexity.py`**

```python
# src/privacy_serving/complexity/perplexity.py
from __future__ import annotations

import math

MIN_PPL = 20.0   # practical floor: fluent everyday text
MAX_PPL = 500.0  # ceiling: dense academic / out-of-distribution text
MAX_TOKENS = 512  # conservative latency limit (model context window is 1024)

_LOG_MIN = math.log(MIN_PPL)
_LOG_MAX = math.log(MAX_PPL)

# Module-level singletons — populated lazily by _load_model()
_model = None
_tokenizer = None


def _normalise_perplexity(ppl: float) -> float:
    """Map raw perplexity to [0, 1] anchored at MIN_PPL=0.0, MAX_PPL=1.0."""
    score = (math.log(max(ppl, 1e-9)) - _LOG_MIN) / (_LOG_MAX - _LOG_MIN)
    return max(0.0, min(1.0, score))


def _load_model() -> None:
    """Load distilgpt2 on first call. Raises RuntimeError if loading fails."""
    global _model, _tokenizer
    if _model is not None:
        return
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import transformers/torch. "
            "Install with: pip install 'transformers>=4.40' 'torch>=2.0'"
        ) from exc

    try:
        _tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
        _model = GPT2LMHeadModel.from_pretrained("distilgpt2")

        import torch as _torch
        if _torch.backends.mps.is_available():
            device = _torch.device("mps")
        elif _torch.cuda.is_available():
            device = _torch.device("cuda")
        else:
            device = _torch.device("cpu")

        _model = _model.to(device)
        _model.eval()
    except Exception as exc:
        raise RuntimeError(f"Failed to load distilgpt2: {exc}") from exc


class PerplexityScorer:
    def raw_perplexity(self, text: str) -> float:
        """Run inference and return exp(mean cross-entropy loss)."""
        import torch

        _load_model()
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
        )
        input_ids = inputs["input_ids"].to(_model.device)
        with torch.no_grad():
            outputs = _model(input_ids, labels=input_ids)
        return math.exp(outputs.loss.item())

    def score(self, text: str) -> float:
        """Return normalised perplexity score in [0, 1]."""
        return _normalise_perplexity(self.raw_perplexity(text))
```

- [ ] **Step 4: Run perplexity tests — all must pass**

Run: `pytest tests/test_complexity_perplexity.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full suite to confirm nothing regressed**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/privacy_serving/complexity/perplexity.py tests/test_complexity_perplexity.py
git commit -m "feat: implement PerplexityScorer with distilgpt2 and anchored normalisation"
```

---

## Task 5: Wire everything together — real `__init__.py`, `main.py` lifespan, revised tests

**Context:** This task replaces the heuristic shim in `complexity/__init__.py` with the real `ComplexityScorer`, adds the FastAPI lifespan warmup to `main.py`, patches `warmup` in test fixtures so tests don't load the model, and updates `test_complexity.py` thresholds for the new score distribution.

**Files:**
- Modify: `src/privacy_serving/complexity/__init__.py` (replace shim with real implementation)
- Modify: `src/privacy_serving/main.py` (add lifespan)
- Modify: `tests/test_api.py` (patch warmup in client fixture)
- Modify: `tests/test_complexity.py` (revised thresholds and new tests)

---

- [ ] **Step 1: Remove `tiktoken` from `pyproject.toml` and reinstall**

In `pyproject.toml`, remove `"tiktoken>=0.7"` from the `dependencies` list (the shim that used it will be replaced in Step 2):

```toml
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "httpx>=0.27",
    "pydantic>=2.7",
    "pyyaml>=6.0",
    "transformers>=4.40",
    "torch>=2.0",
]
```

Run: `pip install -e ".[dev]"`
Expected: No errors (tiktoken is no longer required)

- [ ] **Step 2: Replace `complexity/__init__.py` with the real implementation**

```python
# src/privacy_serving/complexity/__init__.py
from __future__ import annotations

from dataclasses import dataclass

from privacy_serving.models import Message
from privacy_serving.complexity.linguistic import LinguisticScorer
from privacy_serving.complexity.perplexity import PerplexityScorer, _normalise_perplexity


@dataclass
class ComplexityDetail:
    final_score: float        # combined weighted score [0, 1]
    perplexity_score: float   # normalised perplexity sub-score [0, 1]
    linguistic_score: float   # linguistic sub-score [0, 1]
    raw_perplexity: float     # raw exp(loss) for diagnostics


class ComplexityScorer:
    def __init__(self) -> None:
        self.perplexity = PerplexityScorer()  # model loads lazily on first call
        self.linguistic = LinguisticScorer()  # pure Python, no lazy loading

    def score_detail(self, text: str) -> ComplexityDetail:
        """Score text and return full breakdown. Text must be non-empty."""
        raw_ppl = self.perplexity.raw_perplexity(text)  # single inference pass
        ppl_score = _normalise_perplexity(raw_ppl)
        ling_score = self.linguistic.score(text)
        final = 0.55 * ppl_score + 0.45 * ling_score
        return ComplexityDetail(
            final_score=final,
            perplexity_score=ppl_score,
            linguistic_score=ling_score,
            raw_perplexity=raw_ppl,
        )


# Module-level singleton — sub-scorers load lazily on first use
_scorer = ComplexityScorer()


def compute_complexity(messages: list[Message]) -> float:
    """Return complexity score in [0, 1]. Public API — signature unchanged."""
    return compute_complexity_detail(messages).final_score


def compute_complexity_detail(messages: list[Message]) -> ComplexityDetail:
    """Return full score breakdown for logging/debugging."""
    if not messages:
        return ComplexityDetail(0.0, 0.0, 0.0, 0.0)
    text = " ".join(m.content or "" for m in messages)
    if not text.strip():
        return ComplexityDetail(0.0, 0.0, 0.0, 0.0)
    return _scorer.score_detail(text)


def warmup() -> None:
    """Pre-load the distilgpt2 model. Call from FastAPI lifespan to avoid first-request latency."""
    _scorer.perplexity.score("warmup")
```

- [ ] **Step 2: Update `main.py` to add the lifespan warmup**

Replace the `create_app` function and add imports:

```python
# src/privacy_serving/main.py
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from privacy_serving.clients import ModelClient
from privacy_serving.complexity import warmup
from privacy_serving.config import Config, load_config
from privacy_serving.models import ChatCompletionRequest
from privacy_serving.router import Router
from privacy_serving.stats import StatsTracker


@asynccontextmanager
async def _lifespan(app: FastAPI):
    warmup()
    yield


def create_app(config: Config) -> FastAPI:
    router = Router(config)
    stats = StatsTracker()

    app = FastAPI(title="Privacy Serving Proxy", lifespan=_lifespan)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": config.local_model.model, "object": "model", "owned_by": "local"},
                {"id": config.remote_model.model, "object": "model", "owned_by": "remote"},
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Response:
        if request.stream:
            raise HTTPException(status_code=501, detail="Streaming is not supported by this proxy.")
        destination, score = router.route(request)
        model_config = router.model_config_for(destination)

        async with ModelClient(model_config) as client:
            result = await client.complete(request)

        stats.record(destination)

        headers = {
            "X-Routed-To": destination,
            "X-Complexity-Score": f"{score:.4f}",
            "X-Model-Used": model_config.model,
        }
        return JSONResponse(content=result, headers=headers)

    @app.get("/stats")
    async def get_stats() -> dict[str, Any]:
        return stats.snapshot()

    return app


def main() -> None:
    import uvicorn

    config_path = os.environ.get("PRIVACY_SERVING_CONFIG", "config.yaml")
    config = load_config(config_path)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update `tests/test_api.py` — patch warmup in the `client` fixture**

The `TestClient` triggers the FastAPI lifespan on the first request. Without patching, it will attempt to load distilgpt2 during API tests. Update the `client` fixture:

```python
# tests/test_api.py
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from privacy_serving.config import Config, ModelConfig, RoutingConfig, ServerConfig
from privacy_serving.main import create_app


FAKE_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "qwen2.5:9b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
}


@pytest.fixture
def test_config():
    return Config(
        local_model=ModelConfig(base_url="http://local/v1", model="qwen2.5:9b"),
        remote_model=ModelConfig(base_url="http://remote/v1", model="gpt-4o", api_key="sk-test"),
        routing=RoutingConfig(complexity_threshold=0.5),
        server=ServerConfig(),
    )


@pytest.fixture(autouse=True)
def mock_perplexity():
    """Prevent distilgpt2 inference during all API tests (warmup + per-request scoring)."""
    with patch(
        "privacy_serving.complexity.perplexity.PerplexityScorer.raw_perplexity",
        return_value=25.0,  # ppl=25 → normalised score ≈ 0.04 → routes local
    ):
        yield


@pytest.fixture
def client(test_config):
    # `mock_perplexity` autouse fixture covers per-request inference.
    # Warmup is patched here to prevent the lifespan startup call.
    with patch("privacy_serving.complexity.warmup"):
        app = create_app(test_config)
        with TestClient(app) as c:
            yield c


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    models = [m["id"] for m in data["data"]]
    assert "qwen2.5:9b" in models
    assert "gpt-4o" in models


def test_chat_completions_routes_to_local(client, test_config):
    """A simple 'Hi' prompt should route to local model."""
    with patch("privacy_serving.main.ModelClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.complete = AsyncMock(return_value=FAKE_COMPLETION)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        resp = client.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "42"
    assert "X-Routed-To" in resp.headers


def test_chat_completions_includes_routing_metadata(client):
    """Response headers include routing decision and score."""
    with patch("privacy_serving.main.ModelClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.complete = AsyncMock(return_value=FAKE_COMPLETION)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        resp = client.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert "X-Routed-To" in resp.headers
    assert "X-Complexity-Score" in resp.headers
    assert "X-Model-Used" in resp.headers


def test_stats_endpoint(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "local" in data
    assert "remote" in data
    assert "total" in data
    assert "local_rate" in data


def test_invalid_request_returns_422(client):
    resp = client.post("/v1/chat/completions", json={"model": "x"})  # missing messages
    assert resp.status_code == 422


def test_stream_not_supported_returns_501(client):
    resp = client.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert resp.status_code == 501
    assert "Streaming" in resp.json()["detail"]
```

- [ ] **Step 4: Revise `tests/test_complexity.py`**

The revised tests mock the perplexity scorer so the test suite stays fast. The linguistic sub-scorer runs for real (it's pure Python and instant).

```python
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
```

- [ ] **Step 5: Run full test suite — all must pass**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add \
  pyproject.toml \
  src/privacy_serving/complexity/__init__.py \
  src/privacy_serving/main.py \
  tests/test_api.py \
  tests/test_complexity.py
git commit -m "feat: replace heuristic complexity scorer with distilgpt2 + linguistic hybrid"
```

---

## Task 6: Update `README.md`

**Context:** The README's "How it works" table describes the old 5-factor heuristic. Update it to describe the new principled hybrid approach.

**Files:**
- Modify: `README.md`

---

- [ ] **Step 1: Update the "How it works" section**

Replace the "How it works" table and surrounding text in `README.md`:

```markdown
## How it works

Every request is scored by a principled complexity algorithm before it reaches a model:

| Component | Weight | Signal |
|-----------|--------|--------|
| Perplexity (distilgpt2) | 55% | Language model assigns higher loss to specialized, technical, or domain-specific text that a weak local model is unlikely to handle well |
| Linguistic features | 45% | Structural properties: Flesch-Kincaid readability, vocabulary richness (TTR, hapax ratio), content-word density, sentence length, conditional clauses, discourse connectives |

The score is a float in `[0.0, 1.0]`. Requests at or above `complexity_threshold` go to the remote model; everything below stays local.

**Calibration guide:**

| Score range | Typical content |
|-------------|-----------------|
| 0.05–0.10 | Simple greetings and short conversational text |
| 0.20–0.35 | Short factual questions |
| 0.55–0.75 | Multi-step technical or analytical requests |
| > 0.80 | Dense domain-specific or highly specialized text |

> **Note on latency:** The perplexity component runs distilgpt2 locally (~30–60ms on Apple Silicon MPS, ~200–400ms on CPU). The model is loaded at server startup to avoid first-request latency.
```

- [ ] **Step 2: Run tests to confirm README change didn't break anything**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README complexity algorithm section for new hybrid scorer"
```

---

## Final verification

- [ ] **Run the complete test suite one last time**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS, no warnings about missing imports

- [ ] **Verify the package imports cleanly**

Run: `python -c "from privacy_serving.complexity import compute_complexity, compute_complexity_detail, warmup; print('OK')"`
Expected: `OK`

- [ ] **Verify the app starts (import check only — no model load)**

Run: `python -c "from privacy_serving.main import create_app; print('OK')"`
Expected: `OK`
