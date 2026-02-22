"""
wav2vec2-based pronunciation scorer.

Architecture:
  1. facebook/wav2vec2-base encodes raw audio into contextual representations.
  2. The CTC head converts representations → per-frame phoneme log-posteriors.
  3. A forced aligner maps phoneme segments to reference phonemes.
  4. GOP (Goodness of Pronunciation) score is computed per phoneme.
  5. Scores are aggregated to word and utterance level.
"""

import re
import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .base_scorer import (
    BasePronunciationScorer,
    PhonemeResult,
    ScoringResult,
    WordResult,
)
from ..scoring.aggregator import ScoreAggregator

logger = logging.getLogger(__name__)


class Wav2Vec2PronunciationScorer(BasePronunciationScorer):
    """
    Offline pronunciation scorer using wav2vec2 + GOP scoring.

    Recommended model: facebook/wav2vec2-base (95 MB, CPU-friendly)
    Higher accuracy:   facebook/wav2vec2-large-960h (1.18 GB, needs GPU)
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: str | None = None,
        calibration_alpha: float = 6.0,
        calibration_beta: float = -0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s …", model_name, self.device)

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.calibration_alpha = calibration_alpha
        self.calibration_beta = calibration_beta
        self.aggregator = ScoreAggregator()

        # Build vocab: token_id → character/phoneme
        self.vocab = {v: k for k, v in self.processor.tokenizer.get_vocab().items()}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(
        self,
        audio: np.ndarray,
        reference_text: str,
        sample_rate: int = 16000,
    ) -> ScoringResult:
        t0 = time.perf_counter()

        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits  # (1, T, vocab)

        log_probs = F.log_softmax(logits, dim=-1)  # (1, T, vocab)
        log_probs_np = log_probs.squeeze(0).cpu().numpy()  # (T, vocab)

        # Greedy decode to get recognised text
        predicted_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        recognised = self._decode_ctc(predicted_ids)

        # Align reference text to frames and compute GOP
        word_results = self._align_and_score(
            log_probs_np, reference_text, recognised, audio, sample_rate
        )

        result = self.aggregator.aggregate(
            word_results=word_results,
            reference_text=reference_text,
            recognised_text=recognised,
            total_duration_ms=int(len(audio) / sample_rate * 1000),
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Scored in %.1f ms (audio=%.1f s)", latency_ms, len(audio) / sample_rate)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_ctc(self, predicted_ids: list[int]) -> str:
        """Simple greedy CTC decode (collapse blanks and repeated tokens)."""
        blank_id = self.processor.tokenizer.pad_token_id
        tokens = []
        prev = None
        for t in predicted_ids:
            if t != blank_id and t != prev:
                tokens.append(self.vocab.get(t, ""))
            prev = t
        return "".join(tokens).strip().lower()

    def _gop_score(self, log_probs_segment: np.ndarray, phoneme_id: int) -> float:
        """
        GOP = mean log P(correct phoneme | frame) over all frames in segment.
        Mapped to [0, 100] via sigmoid calibration.
        """
        if len(log_probs_segment) == 0:
            return 0.0
        mean_log_prob = float(log_probs_segment[:, phoneme_id].mean())
        # Sigmoid calibration: score = 100 / (1 + exp(-(alpha * x + beta)))
        score = 100.0 / (1.0 + np.exp(-(self.calibration_alpha * mean_log_prob + self.calibration_beta)))
        return float(np.clip(score, 0.0, 100.0))

    def _align_and_score(
        self,
        log_probs: np.ndarray,
        reference_text: str,
        recognised_text: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[WordResult]:
        """
        Lightweight CTC-based forced alignment.

        For production, replace this with Montreal Forced Aligner (MFA) or
        torchaudio's forced_align for higher accuracy.
        """
        words = reference_text.lower().split()
        n_frames = log_probs.shape[0]
        hop_length = 320  # wav2vec2 downsamples 20x from 16 kHz → 50 Hz frames
        ms_per_frame = hop_length / sample_rate * 1000  # ≈ 20 ms

        # Split frames evenly across words (placeholder; replace with MFA)
        frames_per_word = max(1, n_frames // max(len(words), 1))
        recognised_words = recognised_text.split()
        recognised_set = set(recognised_words)

        word_results: list[WordResult] = []
        for i, word in enumerate(words):
            start_frame = i * frames_per_word
            end_frame = min((i + 1) * frames_per_word, n_frames)
            segment = log_probs[start_frame:end_frame]

            offset_ms = int(start_frame * ms_per_frame)
            duration_ms = int((end_frame - start_frame) * ms_per_frame)

            # Determine error type
            if word in recognised_set:
                error_type = "None"
            else:
                error_type = "Mispronunciation"

            # Score phonemes within the word segment
            phoneme_results = self._score_phonemes(segment, word, offset_ms, ms_per_frame)
            word_accuracy = float(np.mean([p.accuracy_score for p in phoneme_results])) if phoneme_results else 0.0

            word_results.append(
                WordResult(
                    word=word,
                    accuracy_score=word_accuracy,
                    error_type=error_type,
                    offset_ms=offset_ms,
                    duration_ms=duration_ms,
                    phonemes=phoneme_results,
                )
            )

        return word_results

    def _score_phonemes(
        self,
        segment: np.ndarray,
        word: str,
        word_offset_ms: int,
        ms_per_frame: float,
    ) -> list[PhonemeResult]:
        """
        Score each phoneme in a word segment.

        Uses the character-level vocab of wav2vec2-base; for true phoneme-level
        scoring fine-tune on TIMIT with phoneme targets or swap to wav2vec2-lv-60-espeak-cv-ft.
        """
        # Build a list of expected characters (graphemes as proxy for phonemes)
        chars = [c for c in word if c.isalpha()]
        if not chars or len(segment) == 0:
            return []

        frames_per_char = max(1, len(segment) // len(chars))
        results: list[PhonemeResult] = []

        for j, char in enumerate(chars):
            char_start = j * frames_per_char
            char_end = min((j + 1) * frames_per_char, len(segment))
            char_segment = segment[char_start:char_end]

            # Look up token id for this character
            token_id = self.processor.tokenizer.convert_tokens_to_ids(char.upper())
            if token_id is None or token_id == self.processor.tokenizer.unk_token_id:
                accuracy = 50.0  # unknown char → neutral score
            else:
                accuracy = self._gop_score(char_segment, token_id)

            offset_ms = word_offset_ms + int(char_start * ms_per_frame)
            duration_ms = int((char_end - char_start) * ms_per_frame)

            results.append(
                PhonemeResult(
                    phoneme=char,
                    accuracy_score=accuracy,
                    offset_ms=offset_ms,
                    duration_ms=duration_ms,
                )
            )

        return results
