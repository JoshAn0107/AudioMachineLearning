"""
Fluency score estimation.

Azure defines FluencyScore as the degree to which the speech matches
a native speaker's use of silent breaks. We approximate this by:
  1. Measuring speaking rate (syllables or words per second)
  2. Penalising gaps between words that are unusually long
  3. Returning a 0-100 score.

Adjustments for short utterances:
  - Single words: fluency = 100 (no inter-word pacing to measure)
  - 2 words: reduced rate penalty (lower WPS target)
"""

from __future__ import annotations

import numpy as np

from ..models.base_scorer import WordResult


# Native English: ~130-160 words per minute -> 2.2-2.7 wps
NATIVE_WPS_TARGET = 2.4
NATIVE_WPS_STD = 0.5
MAX_PAUSE_MS = 500        # pauses > 500 ms are penalised


def estimate_fluency_score(
    word_results: list[WordResult],
    total_duration_ms: int,
    reference_text: str,
) -> float:
    if not word_results or total_duration_ms <= 0:
        return 0.0

    n_words = len(word_results)

    # Single words: fluency is not measurable (no inter-word pacing).
    # Azure returns 100 for single words; we match that behavior.
    if n_words == 1:
        return 100.0

    duration_sec = total_duration_ms / 1000.0
    wps = n_words / duration_sec if duration_sec > 0 else 0.0

    # Adjust WPS target for short utterances (2-3 words are naturally slower)
    if n_words <= 2:
        target_wps = 1.5
        std = 0.6
    elif n_words <= 4:
        target_wps = 2.0
        std = 0.5
    else:
        target_wps = NATIVE_WPS_TARGET
        std = NATIVE_WPS_STD

    # Rate score: penalise deviation from target
    rate_z = abs(wps - target_wps) / std
    rate_score = max(0.0, 100.0 - 20.0 * rate_z)

    # Pause score: penalise long gaps between words
    pause_penalties = []
    for i in range(1, len(word_results)):
        prev_end = word_results[i - 1].offset_ms + word_results[i - 1].duration_ms
        curr_start = word_results[i].offset_ms
        gap_ms = max(0, curr_start - prev_end)
        if gap_ms > MAX_PAUSE_MS:
            penalty = min(30.0, (gap_ms - MAX_PAUSE_MS) / 100.0 * 5.0)
            pause_penalties.append(penalty)

    pause_score = max(0.0, 100.0 - sum(pause_penalties))

    return float(np.clip(0.6 * rate_score + 0.4 * pause_score, 0.0, 100.0))
