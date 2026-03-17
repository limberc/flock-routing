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

No other module changes. The `Router`, `main.py`, and all tests that call `compute_complexity` continue to work without modification. Note: existing threshold assertions in `tests/test_complexity.py` (e.g. `< 0.2`, `>= 0.4`) will be revised to match the new scorer's absolute score distribution.

New dependencies added to `pyproject.toml`:
- `transformers>=4.40`
- `torch>=2.0`

---

## Component 1: PerplexityScorer (`perplexity.py`)

### Approach

Use `distilgpt2` to compute the mean cross-entropy loss over the input tokens. A general-purpose LM assigns low loss to everyday language and high loss to specialized, technical, or domain-specific text — exactly the signal needed to distinguish prompts a weak local model can handle from those it cannot.

### Implementation

- Model (`GPT2LMHeadModel`) and tokenizer (`GPT2TokenizerFast`) are module-level singletons, loaded lazily on first call via a `_load_model()` helper
- Device selection: use MPS if available (Apple Silicon), else CUDA if available, else CPU
- Model is set to `eval()` mode; all inference runs under `torch.no_grad()`
- Input is truncated to **512 tokens** as a latency-optimized limit (distilgpt2's actual context window is 1024 tokens; 512 is a deliberate conservative choice for speed)
- Raw perplexity = `exp(mean cross-entropy loss)`

**Normalisation** — anchored to the practical floor of distilgpt2 (~20 ppl for fluent English):

```
MIN_PPL = 20.0   # floor: fluent everyday text
MAX_PPL = 500.0  # ceiling: dense academic / highly specialized domain text

score = clamp((log(ppl) - log(MIN_PPL)) / (log(MAX_PPL) - log(MIN_PPL)), 0.0, 1.0)
```

Calibration table:

| ppl | score | Example |
|-----|-------|---------|
| 20 | 0.00 | Simple conversational text |
| 50 | 0.28 | Technical documentation |
| 150 | 0.63 | Specialized academic text |
| 500 | 1.00 | Dense domain-specific / out-of-distribution text |

All log operations use natural log (`math.log`).

- If the model fails to load, raise `RuntimeError` with a clear message on first call
- **Warmup:** `__init__.py` exposes a module-level `warmup()` function that calls `_scorer.perplexity.score("warmup")`. Wire it into `main.py` via a FastAPI lifespan context manager so the model loads at startup rather than on the first live request:
  ```python
  from contextlib import asynccontextmanager
  from privacy_serving.complexity import warmup

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      warmup()
      yield

  app = FastAPI(lifespan=lifespan)
  ```
  In tests that construct the app via `create_app()`, patch `privacy_serving.complexity.warmup` to a no-op to avoid loading the model during test collection:
  ```python
  with patch("privacy_serving.complexity.warmup"):
      app = create_app(config)
  ```

**Latency note:** On Apple Silicon (MPS), distilgpt2 at 512 tokens runs in ~30–60ms. On CPU it may reach 200–400ms — consider setting `MAX_TOKENS = 256` for CPU-only deployments if latency is a concern.

### Interface

```python
class PerplexityScorer:
    def score(self, text: str) -> float: ...          # normalised [0, 1]
    def raw_perplexity(self, text: str) -> float: ... # raw exp(loss); runs inference independently
```

`score()` and `raw_perplexity()` are independent methods that each run inference. When `compute_complexity_detail` needs both, it calls `raw_perplexity()` once and derives the normalised score from that result, to avoid two inference passes. The combining layer in `__init__.py` is responsible for this; `PerplexityScorer.score()` is a convenience wrapper that calls `raw_perplexity()` internally and applies the normalisation formula.

---

## Component 2: LinguisticScorer (`linguistic.py`)

### Approach

Seven structural text features, each measuring a property of the text itself — no topic keywords, no domain lists. Pure Python, no NLTK or spaCy required.

### Text preprocessing

- Words: `re.findall(r"[a-zA-Z']+", text)` — ASCII-only; Unicode characters (Greek letters, subscripts) are silently dropped, which is an accepted limitation
- Sentences: split via `re.split(r'[.?!]', text)`, then keep only segments with ≥ 3 characters; `sentence_count = max(1, len(detected_sentences))` to prevent ZeroDivisionError
- `total_words = max(1, len(words))` to prevent ZeroDivisionError

### Features

| Feature | Formula | Normalisation | Weight |
|---|---|---|---|
| Flesch-Kincaid Grade Level | `0.39 × (W/S) + 11.8 × (Syl/W) − 15.59` | `clamp(max(0, grade) / 18.0, 0, 1)` | 20% |
| Type-Token Ratio | `unique_words / total_words` | `clamp(ttr / 0.9, 0, 1)`; 0.0 if `total_words < 5` | 15% |
| Hapax ratio | `once-occurring_words / total_words` | `clamp(hapax / 0.7, 0, 1)`; 0.0 if `total_words < 5` | 15% |
| Content-word density | `content_words / total_words` | `clamp(density / 0.8, 0, 1)` | 15% |
| Average sentence length | `total_words / sentence_count` | `clamp(avg_len / 40.0, 0, 1)` | 15% |
| Conditional density | `conditional_hits / sentence_count` | `clamp(density / 2.0, 0, 1)` | 10% |
| Discourse connective density | `connective_hits / sentence_count` | `clamp(density / 1.5, 0, 1)` | 10% |

Note: TTR and hapax ratio return 0.0 when `total_words < 5` to avoid inflated scores on trivially short inputs where all words are trivially unique.

**Syllable counting** (explicit algorithm — intentionally approximate; sufficient for FK grade level estimation):

```python
def _count_syllables(word: str) -> int:
    word = word.lower()
    count = len(re.findall(r'[aeiou]+', word))   # count vowel groups
    if word.endswith('e') and len(word) > 2 and count >= 2:
        count -= 1                                # silent-e adjustment
    return max(1, count)
```

Known limitation: words ending in "le" (e.g., "simple", "able") are underestimated by one syllable. This is accepted — the algorithm is a heuristic approximation, not a linguistically precise syllabifier. Implementers must not modify this algorithm to fix edge cases, as any variant will have different edge cases and would break calibration.

**Marker detection** — multi-word markers are detected via `text.lower().count(marker)` (substring match). The single-character marker `if` must use word-boundary matching — `len(re.findall(r'\bif\b', text_lower))` — to avoid false positives inside words like "wifi", "tariff", "notify", "specify". All other markers use `.count()`.

- Conditional markers: `if` (word-boundary), `unless`, `when`, `provided`, `assuming`, `whether`, `given that`, `in case`, `as long as`
- Discourse connectives: `however`, `therefore`, `thus`, `consequently`, `hence`, `moreover`, `furthermore`, `nevertheless`, `nonetheless`, `although`, `whereas`

**Content-word stopword set** (hardcoded as a `frozenset`, 130 unique words — deduplicated):

```python
STOPWORDS = frozenset({
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
```

### Interface

```python
class LinguisticScorer:
    def score(self, text: str) -> float: ...                   # weighted average [0, 1]
    def features(self, text: str) -> dict[str, float]: ...     # per-feature normalised breakdown
```

Empty input (`text == ""`) returns `score = 0.0` and all features `0.0`.

---

## Component 3: Combining (`__init__.py`)

### Weights

| Sub-scorer | Weight | Rationale |
|---|---|---|
| Perplexity score | 55% | Strongest single signal; catches domain specialization no rule can |
| Linguistic score | 45% | Catches task complexity (multi-step, dense constraints) that perplexity misses on well-formed text |

### `ComplexityScorer` class

```python
class ComplexityScorer:
    def __init__(self) -> None:
        self.perplexity = PerplexityScorer()   # lazy — model loads on first call
        self.linguistic = LinguisticScorer()   # no lazy loading needed (pure Python)

    def score_detail(self, text: str) -> ComplexityDetail:
        raw_ppl = self.perplexity.raw_perplexity(text)  # single inference pass
        ppl_score = _normalise_perplexity(raw_ppl)       # apply normalisation formula
        ling_score = self.linguistic.score(text)
        final = 0.55 * ppl_score + 0.45 * ling_score
        return ComplexityDetail(
            final_score=final,
            perplexity_score=ppl_score,
            linguistic_score=ling_score,
            raw_perplexity=raw_ppl,
        )
```

`_normalise_perplexity(ppl)` applies the `clamp((log(ppl) - log(MIN_PPL)) / (log(MAX_PPL) - log(MIN_PPL)), 0.0, 1.0)` formula.

### Public API

```python
from dataclasses import dataclass

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

def warmup() -> None:
    """Pre-load the perplexity model. Call from FastAPI lifespan startup."""
```

- Text extraction: `" ".join(m.content or "" for m in messages)`
- Empty messages list → `ComplexityDetail(0.0, 0.0, 0.0, 0.0)`
- `_scorer = ComplexityScorer()` is a module-level singleton instantiated at import time; the `PerplexityScorer` sub-scorer loads lazily on first use
- `warmup()` calls `_scorer.perplexity.score("warmup")` to trigger lazy loading at startup

**Threshold guidance:** The default `complexity_threshold: 0.5` in `config.yaml` remains appropriate for the new scorer. Under the new distribution: "Hi" scores ~0.05–0.10, a short factual question scores ~0.20–0.35, a multi-step technical request scores ~0.55–0.75. Operators using the old heuristic scorer with a custom threshold may want to re-evaluate against their own traffic.

---

## Testing

### `tests/test_complexity_linguistic.py`
- Each of the 7 features in isolation
- Edge cases: empty string, single word, single sentence, no punctuation (ZeroDivisionError guard)
- TTR/hapax return 0.0 for inputs under 5 words
- FK grade clamped to 0.0 minimum (no negative values)
- Score bounded [0, 1] for all inputs

### `tests/test_complexity_perplexity.py`
- Score ordering: technical prompt scores higher than casual prompt
- Score bounded [0, 1]
- `raw_perplexity` > 1.0 for any real input
- Model loading can be patched with `unittest.mock.patch` for fast CI

### `tests/test_complexity.py` (revised)
- All existing tests retained but thresholds revised to match new scorer distribution
- Trivial prompt ("Hi") → score < 0.20
- Short factual → score < 0.45
- Complex reasoning prompt → score > 0.55
- Many questions > single question (relative)
- Score bounded [0, 1]
- Empty messages → 0.0
- Message-count ordering test dropped (message count is no longer a signal; content is)

---

## Migration

1. Delete `src/privacy_serving/complexity.py`
2. Create `src/privacy_serving/complexity/` package as specified
3. In `pyproject.toml`: add `transformers>=4.40` and `torch>=2.0` to `[project] dependencies`; remove `tiktoken` (no longer used after `complexity.py` is deleted)
4. Add a FastAPI `lifespan` context manager in `main.py` that calls `warmup()` on startup (see warmup section above); patch `privacy_serving.complexity.warmup` in integration tests that call `create_app()` to prevent model loading during tests
5. Revise threshold assertions in `tests/test_complexity.py`
6. Update `README.md` complexity algorithm section

---

## Non-Goals

- Streaming support (already rejected at the API level)
- Online learning / updating weights from routing outcomes
- Per-domain threshold tuning
- Caching perplexity scores across requests
- Unicode / non-ASCII text support in linguistic scorer
