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
