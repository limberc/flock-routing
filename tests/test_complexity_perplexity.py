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
