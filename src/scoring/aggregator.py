"""
Score aggregation: phoneme -> word -> utterance.

Produces scores that match Azure's PronunciationAssessment fields:
  - AccuracyScore   - how closely phonemes match native pronunciation
  - FluencyScore    - penalizes pauses and hesitation
  - CompletenessScore - fraction of reference words that were pronounced
  - PronScore       - overall weighted score (Azure weights: 0.4/0.2/0.4)
"""

from __future__ import annotations

import numpy as np

from .fluency import estimate_fluency_score
from ..models.base_scorer import ScoringResult, WordResult


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two words are a fuzzy match (handles minor pronunciation variants)."""
    if a == b:
        return True
    # Allow 1-char difference for words > 3 chars (e.g., hello vs hallo)
    if len(a) > 3 and len(b) > 3:
        if abs(len(a) - len(b)) <= 1:
            diffs = sum(1 for x, y in zip(a, b) if x != y)
            extra = abs(len(a) - len(b))
            if diffs + extra <= 1:
                return True
    # Check if one contains the other (compound words: woodchuck vs wood+chuck)
    if a in b or b in a:
        return True
    return False


class ScoreAggregator:
    """
    Combines per-word/phoneme scores into utterance-level scores
    following Azure's weighting scheme.
    """

    # Azure's documented weights for PronScore
    ACCURACY_WEIGHT = 0.4
    FLUENCY_WEIGHT = 0.2
    COMPLETENESS_WEIGHT = 0.4

    def aggregate(
        self,
        word_results: list[WordResult],
        reference_text: str,
        recognised_text: str,
        total_duration_ms: int,
    ) -> ScoringResult:
        reference_words = reference_text.lower().split()
        recognised_words = recognised_text.lower().split()

        # Accuracy: mean of per-word accuracy scores (weighted by word length)
        if word_results:
            weights = np.array([max(1, len(w.word)) for w in word_results], dtype=float)
            accuracy_scores = np.array([w.accuracy_score for w in word_results])
            accuracy_score = float(np.average(accuracy_scores, weights=weights))
        else:
            accuracy_score = 0.0

        # Completeness: fraction of reference words matched in recognised output
        # Uses fuzzy matching to handle minor pronunciation variants
        if reference_words:
            matched = 0
            used_rec = set()
            for ref_w in reference_words:
                for j, rec_w in enumerate(recognised_words):
                    if j not in used_rec and _fuzzy_match(ref_w, rec_w):
                        matched += 1
                        used_rec.add(j)
                        break
            completeness_score = min(100.0, 100.0 * matched / len(reference_words))
        else:
            completeness_score = 100.0

        # Fluency: estimated from timing characteristics
        fluency_score = estimate_fluency_score(
            word_results=word_results,
            total_duration_ms=total_duration_ms,
            reference_text=reference_text,
        )

        # PronScore: Azure's weighted combination
        pron_score = (
            self.ACCURACY_WEIGHT * accuracy_score
            + self.FLUENCY_WEIGHT * fluency_score
            + self.COMPLETENESS_WEIGHT * completeness_score
        )

        recognition_status = "Success" if recognised_text.strip() else "NoMatch"
        return ScoringResult(
            recognition_status=recognition_status,
            display_text=recognised_text.title() if recognised_text else reference_text,
            offset_ms=0,
            duration_ms=total_duration_ms,
            pron_score=round(pron_score, 2),
            accuracy_score=round(accuracy_score, 2),
            fluency_score=round(fluency_score, 2),
            completeness_score=round(completeness_score, 2),
            words=word_results,
        )
