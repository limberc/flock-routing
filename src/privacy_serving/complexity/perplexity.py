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
