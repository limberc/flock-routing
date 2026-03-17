# Smart Complexity Scorer Design

## Problem

The current `compute_complexity` implementation is driven entirely by ad-hoc pre-defined rules: keyword lists, token counts, and question-mark counting. These rules are brittle — a highly complex query that does not match any listed keyword will be scored as simple, and the lists require continuous manual maintenance as new domains emerge.

## Goal

Replace the heuristic scorer with a principled, generalizable algorithm that measures intrinsic linguistic and cognitive complexity without encoding domain knowledge as static lists. The public API (`compute_complexity(messages) -> float`) must remain unchanged.

## Architecture

`src/privacy_serving/complexity.py` is replaced by a `src/privacy_serving/complexity/` package:

```
src/privacy_serving/complexity/
    __init__.py       # exports compute_complexity, compute_complexity_detail
    perplexity.py     # PerplexityScorer using distilgpt2
    linguistic.py     # LinguisticScorer using pure-Python text features
```

No other module changes. The `Router`, `main.py`, and all tests that call `compute_complexity` continue to work without modification.

New dependency: `transformers>=4.40` (PyTorch already present in the environment).

---

## Component 1: PerplexityScorer (`perplexity.py`)

### Approach

Use `distilgpt2` to compute the mean cross-entropy loss over the input tokens. A general-purpose LM assigns low loss to everyday language and high loss to specialized, technical, or domain-specific text — exactly the signal needed to distinguish prompts a weak local model can handle from those it cannot.

### Implementation

- Model (`GPT2LMHeadModel`) and tokenizer (`GPT2TokenizerFast`) are module-level singletons, loaded lazily on first call via a `_load_model()` helper
- Model is set to `eval()` mode; all inference runs under `torch.no_grad()`
- Input is truncated to **512 tokens** (distilgpt2's context window); this is sufficient for routing decisions
- Raw perplexity = `exp(mean cross-entropy loss)`
- Normalisation: `min(log(ppl) / log(500), 1.0)` using natural log
  - ppl ≈ 20–50 → score ≈ 0.0–0.3 (conversational text)
  - ppl ≈ 50–150 → score ≈ 0.3–0.6 (technical documentation)
  - ppl ≈ 150–500 → score ≈ 0.6–1.0 (specialized academic / domain text)
- If the model fails to load, raise `RuntimeError` immediately at first call with a clear message; do not silently fall back

### Interface

```python
class PerplexityScorer:
    def score(self, text: str) -> float: ...   # returns normalised [0, 1]
    def raw_perplexity(self, text: str) -> float: ...   # returns raw exp(loss)
```

---

## Component 2: LinguisticScorer (`linguistic.py`)

### Approach

Seven structural text features, each measuring a property of the text itself — no topic keywords, no domain lists. Pure Python, no NLTK or spaCy required.

### Features

| Feature | Formula | Saturation point | Weight |
|---|---|---|---|
| Flesch-Kincaid Grade Level | `0.39 × (W/S) + 11.8 × (Syl/W) − 15.59` | 18 | 20% |
| Type-Token Ratio | `unique_words / total_words` | 0.9 | 15% |
| Hapax ratio | `once-occurring words / total_words` | 0.7 | 15% |
| Content-word density | `content_words / total_words` | 0.8 | 15% |
| Average sentence length | `total_words / sentence_count` | 40 words | 15% |
| Conditional density | `conditional_markers / sentence_count` | 2.0 | 10% |
| Discourse connective density | `connective_markers / sentence_count` | 1.5 | 10% |

**Implementation notes:**
- Syllables: vowel-group counting heuristic (`re.findall(r'[aeiou]+', word)`), adjusted for silent-e, minimum 1
- Sentences: split on `.`, `?`, `!` with length ≥ 3 characters
- Words: `re.findall(r"[a-zA-Z']+", text)`
- Content words: words not in a hardcoded ~150-word stopword set (articles, prepositions, pronouns, auxiliaries, conjunctions)
- Conditional markers: `if, unless, when, provided, assuming, whether, given that, in case, as long as`
- Discourse connectives: `however, therefore, thus, consequently, hence, moreover, furthermore, nevertheless, nonetheless, although, whereas`
- Each feature is individually clamped to [0, 1] before weighting: `min(raw_value / saturation_point, 1.0)`
- Empty input returns 0.0 for all features

### Interface

```python
class LinguisticScorer:
    def score(self, text: str) -> float: ...          # weighted average [0, 1]
    def features(self, text: str) -> dict[str, float]: ...  # per-feature breakdown
```

---

## Component 3: Combining (`__init__.py`)

### Weights

| Sub-scorer | Weight | Rationale |
|---|---|---|
| Perplexity score | 55% | Strongest single signal; catches domain specialization no rule can |
| Linguistic score | 45% | Catches task complexity (multi-step, dense constraints) that perplexity misses on well-formed text |

### Public API

```python
@dataclass
class ComplexityDetail:
    final_score: float          # combined [0, 1]
    perplexity_score: float     # normalised perplexity sub-score [0, 1]
    linguistic_score: float     # linguistic sub-score [0, 1]
    raw_perplexity: float       # raw exp(loss) for diagnostics

def compute_complexity(messages: list[Message]) -> float:
    """Unchanged public API — returns final_score only."""

def compute_complexity_detail(messages: list[Message]) -> ComplexityDetail:
    """Returns full breakdown for logging/debugging."""
```

- Both functions extract text from messages using `" ".join(m.content or "" for m in messages)`
- Empty messages list → `ComplexityDetail(0.0, 0.0, 0.0, 0.0)`
- `ComplexityScorer` is a module-level singleton instantiated at import time; sub-scorers initialise lazily

---

## Testing

| Test file | Coverage |
|---|---|
| `tests/test_complexity_linguistic.py` | Each linguistic feature in isolation; edge cases (empty, single word, single sentence) |
| `tests/test_complexity_perplexity.py` | Score ordering (technical > conversational); normalisation bounds; model loading |
| `tests/test_complexity.py` | All existing tests must continue to pass; add combined scorer ordering tests |

Perplexity tests may use `unittest.mock.patch` to avoid loading the model in CI.

---

## Migration

1. Delete `src/privacy_serving/complexity.py`
2. Create `src/privacy_serving/complexity/` package as specified
3. Add `transformers>=4.40` to `pyproject.toml` dependencies
4. All existing tests pass without modification
5. Update `README.md` complexity algorithm section

---

## Non-Goals

- Streaming support (already rejected at the API level)
- Online learning / updating weights from routing outcomes
- Per-domain threshold tuning
- Caching perplexity scores across requests
