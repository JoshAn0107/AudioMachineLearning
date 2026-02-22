"""
Classic GOP (Goodness of Pronunciation) scorer.

Reference:
  Witt & Young (2000). "Phone-level pronunciation scoring and assessment for
  interactive language learning." Speech Communication.

Formula:
  GOP(p) = |log P(p | o_s:e)| / (e - s)

where P(p | o) is the acoustic model posterior for phoneme p given observation o,
and s, e are the start and end frames from forced alignment.

This module wraps any acoustic model that outputs frame-level phoneme log-posteriors
and combines it with forced alignment output.
"""

from __future__ import annotations

import numpy as np

from .base_scorer import (
    BasePronunciationScorer,
    PhonemeResult,
    ScoringResult,
    WordResult,
)
from ..scoring.aggregator import ScoreAggregator


class Alignment:
    """Forced alignment result for a single utterance."""

    def __init__(self, segments: list[dict]):
        """
        Args:
            segments: list of dicts with keys:
                - phoneme (str): IPA or ARPAbet symbol
                - start_frame (int)
                - end_frame (int)
                - word (str): parent word
                - word_start_frame (int)
                - word_end_frame (int)
        """
        self.segments = segments


class GOPScorer(BasePronunciationScorer):
    """
    GOP scorer that takes an external forced alignment and frame-level posteriors.

    Usage:
        aligner = MontrealForcedAligner(...)  # or torchaudio forced_align
        acoustic_model = Wav2Vec2AcousticModel(...)

        scorer = GOPScorer(acoustic_model, phoneme_vocab)
        result = scorer.score(audio, reference_text)
    """

    def __init__(
        self,
        acoustic_model,          # object with .get_log_posteriors(audio, sr) -> np.ndarray (T, V)
        phoneme_vocab: dict,     # phoneme_str -> vocab_index
        aligner=None,            # object with .align(audio, text, sr) -> Alignment
        calibration_alpha: float = 6.0,
        calibration_beta: float = -0.5,
    ):
        self.acoustic_model = acoustic_model
        self.phoneme_vocab = phoneme_vocab
        self.aligner = aligner
        self.calibration_alpha = calibration_alpha
        self.calibration_beta = calibration_beta
        self.aggregator = ScoreAggregator()

    def score(
        self,
        audio: np.ndarray,
        reference_text: str,
        sample_rate: int = 16000,
    ) -> ScoringResult:
        log_posteriors = self.acoustic_model.get_log_posteriors(audio, sample_rate)

        if self.aligner is not None:
            alignment = self.aligner.align(audio, reference_text, sample_rate)
        else:
            alignment = self._uniform_alignment(reference_text, len(log_posteriors))

        word_results = self._score_from_alignment(log_posteriors, alignment)

        return self.aggregator.aggregate(
            word_results=word_results,
            reference_text=reference_text,
            recognised_text=" ".join(w.word for w in word_results),
            total_duration_ms=int(len(audio) / sample_rate * 1000),
        )

    # ------------------------------------------------------------------

    def gop(self, log_posteriors: np.ndarray, phoneme_id: int) -> float:
        """
        Compute GOP score for a segment and return a calibrated [0, 100] value.

        Raw GOP = mean log-posterior of the correct phoneme over the segment.
        """
        if len(log_posteriors) == 0:
            return 0.0
        raw = float(log_posteriors[:, phoneme_id].mean())
        calibrated = 100.0 / (
            1.0 + np.exp(-(self.calibration_alpha * raw + self.calibration_beta))
        )
        return float(np.clip(calibrated, 0.0, 100.0))

    def _score_from_alignment(
        self,
        log_posteriors: np.ndarray,
        alignment: Alignment,
        ms_per_frame: float = 20.0,
    ) -> list[WordResult]:
        """Build WordResult list from phoneme alignment segments."""
        # Group segments by word
        words: dict[str, list] = {}
        for seg in alignment.segments:
            key = (seg["word"], seg["word_start_frame"])
            words.setdefault(key, []).append(seg)

        word_results: list[WordResult] = []
        for (word_str, word_start_frame), segs in sorted(words.items(), key=lambda x: x[0][1]):
            phoneme_results: list[PhonemeResult] = []
            for seg in segs:
                phoneme_id = self.phoneme_vocab.get(seg["phoneme"], -1)
                segment_frames = log_posteriors[seg["start_frame"]: seg["end_frame"]]

                if phoneme_id >= 0:
                    acc = self.gop(segment_frames, phoneme_id)
                else:
                    acc = 50.0  # unknown phoneme → neutral

                phoneme_results.append(
                    PhonemeResult(
                        phoneme=seg["phoneme"],
                        accuracy_score=acc,
                        offset_ms=int(seg["start_frame"] * ms_per_frame),
                        duration_ms=int((seg["end_frame"] - seg["start_frame"]) * ms_per_frame),
                    )
                )

            word_end_frame = segs[-1]["word_end_frame"]
            word_accuracy = np.mean([p.accuracy_score for p in phoneme_results]) if phoneme_results else 0.0
            error_type = "None" if word_accuracy >= 60.0 else "Mispronunciation"

            word_results.append(
                WordResult(
                    word=word_str,
                    accuracy_score=float(word_accuracy),
                    error_type=error_type,
                    offset_ms=int(word_start_frame * ms_per_frame),
                    duration_ms=int((word_end_frame - word_start_frame) * ms_per_frame),
                    phonemes=phoneme_results,
                )
            )

        return word_results

    @staticmethod
    def _uniform_alignment(reference_text: str, n_frames: int) -> Alignment:
        """Fallback: split frames uniformly across words and characters."""
        words = reference_text.lower().split()
        frames_per_word = max(1, n_frames // max(len(words), 1))
        segments = []
        for i, word in enumerate(words):
            w_start = i * frames_per_word
            w_end = min((i + 1) * frames_per_word, n_frames)
            chars = [c for c in word if c.isalpha()]
            frames_per_char = max(1, (w_end - w_start) // max(len(chars), 1))
            for j, char in enumerate(chars):
                segments.append(
                    {
                        "phoneme": char,
                        "start_frame": w_start + j * frames_per_char,
                        "end_frame": min(w_start + (j + 1) * frames_per_char, w_end),
                        "word": word,
                        "word_start_frame": w_start,
                        "word_end_frame": w_end,
                    }
                )
        return Alignment(segments)
