"""Abstract base class for all pronunciation scorers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PhonemeResult:
    phoneme: str          # IPA symbol
    accuracy_score: float # 0–100
    offset_ms: int        # start time in milliseconds
    duration_ms: int      # duration in milliseconds


@dataclass
class WordResult:
    word: str
    accuracy_score: float      # 0–100
    error_type: str            # "None" | "Omission" | "Insertion" | "Mispronunciation"
    offset_ms: int
    duration_ms: int
    phonemes: list[PhonemeResult]


@dataclass
class ScoringResult:
    """Top-level result matching Azure Pronunciation Assessment structure."""
    recognition_status: str    # "Success" | "NoMatch" | "InitialSilenceTimeout"
    display_text: str
    offset_ms: int
    duration_ms: int
    pron_score: float          # overall weighted score
    accuracy_score: float
    fluency_score: float
    completeness_score: float
    words: list[WordResult]

    def to_azure_format(self) -> dict:
        """Serialize to a dict that mirrors Azure's NBest response schema."""
        return {
            "RecognitionStatus": self.recognition_status,
            "DisplayText": self.display_text,
            "Offset": self.offset_ms * 10_000,   # Azure uses 100-ns ticks
            "Duration": self.duration_ms * 10_000,
            "NBest": [
                {
                    "Confidence": 1.0,
                    "Lexical": self.display_text.lower(),
                    "ITN": self.display_text.lower(),
                    "MaskedITN": self.display_text.lower(),
                    "Display": self.display_text,
                    "PronScore": round(self.pron_score, 2),
                    "AccuracyScore": round(self.accuracy_score, 2),
                    "FluencyScore": round(self.fluency_score, 2),
                    "CompletenessScore": round(self.completeness_score, 2),
                    "Words": [
                        {
                            "Word": w.word,
                            "AccuracyScore": round(w.accuracy_score, 2),
                            "ErrorType": w.error_type,
                            "Offset": w.offset_ms * 10_000,
                            "Duration": w.duration_ms * 10_000,
                            "Phonemes": [
                                {
                                    "Phoneme": p.phoneme,
                                    "AccuracyScore": round(p.accuracy_score, 2),
                                    "Offset": p.offset_ms * 10_000,
                                    "Duration": p.duration_ms * 10_000,
                                }
                                for p in w.phonemes
                            ],
                        }
                        for w in self.words
                    ],
                }
            ],
        }


class BasePronunciationScorer(ABC):
    """All pronunciation scorers implement this interface."""

    @abstractmethod
    def score(
        self,
        audio: np.ndarray,
        reference_text: str,
        sample_rate: int = 16000,
    ) -> ScoringResult:
        """
        Score the pronunciation of `audio` against `reference_text`.

        Args:
            audio: Raw PCM waveform, float32, mono.
            reference_text: The word or sentence the speaker intended to say.
            sample_rate: Audio sample rate (default 16 kHz).

        Returns:
            ScoringResult with Azure-compatible scores.
        """
        ...
