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
