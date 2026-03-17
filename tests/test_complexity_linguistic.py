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
