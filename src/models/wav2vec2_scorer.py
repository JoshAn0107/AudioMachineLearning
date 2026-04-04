"""
wav2vec2-based pronunciation scorer.

Architecture:
  1. wav2vec2 encodes raw audio into contextual representations.
  2. The CTC head converts representations → per-frame character log-posteriors.
  3. CTC decoding identifies which frames correspond to which characters.
  4. GOP (Goodness of Pronunciation) score uses aligned frame regions.
  5. Scores are aggregated to word and utterance level.
"""

# pyright: reportMissingImports=false

import re
import time
import logging
from pathlib import Path
from typing import TypedDict

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


class AlignmentEntry(TypedDict):
    token_id: int
    char: str
    start_frame: int
    end_frame: int


class RecognizedWord(TypedDict):
    text: str
    chars: list[AlignmentEntry]
    start_frame: int
    end_frame: int


class Wav2Vec2PronunciationScorer(BasePronunciationScorer):
    """
    Offline pronunciation scorer using wav2vec2 + CTC-aligned GOP scoring.

    Uses CTC decoded frame positions for alignment instead of uniform
    splitting, producing far more accurate phoneme-level scores.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: str | None = None,
        calibration_alpha: float = 3.0,
        calibration_beta: float = 3.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s …", model_name, self.device)

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.calibration_alpha = calibration_alpha
        self.calibration_beta = calibration_beta
        self.aggregator = ScoreAggregator()

        # Build vocab: token_id → character
        self.vocab = {v: k for k, v in self.processor.tokenizer.get_vocab().items()}
        self.blank_id = self.processor.tokenizer.pad_token_id
        self.word_sep_id = self.processor.tokenizer.convert_tokens_to_ids("|")

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

        # Greedy decode to get recognised text and frame alignment
        predicted_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        recognised, frame_alignment = self._decode_ctc_with_alignment(predicted_ids)

        # Score using CTC-aligned frame positions
        word_results = self._align_and_score(
            log_probs_np, reference_text, recognised, frame_alignment, audio, sample_rate
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
    # CTC decoding with frame alignment
    # ------------------------------------------------------------------

    def _decode_ctc_with_alignment(
        self, predicted_ids: list[int]
    ) -> tuple[str, list[AlignmentEntry]]:
        """
        CTC greedy decode that tracks which frames correspond to each token.

        Returns:
            recognised: decoded text string
            frame_alignment: list of {token_id, char, start_frame, end_frame}
        """
        alignment = []
        prev = None

        for frame_idx, tid in enumerate(predicted_ids):
            if tid == self.blank_id:
                prev = tid
                continue
            if tid == prev:
                # Extend duration of current token
                if alignment:
                    alignment[-1]["end_frame"] = frame_idx + 1
                prev = tid
                continue

            char = self.vocab.get(tid, "")
            alignment.append({
                "token_id": tid,
                "char": char,
                "start_frame": frame_idx,
                "end_frame": frame_idx + 1,
            })
            prev = tid

        # Build text from alignment — replace | (word separator) with space
        raw = "".join(a["char"] for a in alignment).strip().lower()
        text = raw.replace("|", " ").strip()
        # Collapse multiple spaces
        text = " ".join(text.split())
        return text, alignment

    def _decode_ctc(self, predicted_ids: list[int]) -> str:
        """Simple greedy CTC decode (backward compat)."""
        text, _ = self._decode_ctc_with_alignment(predicted_ids)
        return text

    # ------------------------------------------------------------------
    # GOP scoring
    # ------------------------------------------------------------------

    def _gop_score(self, log_probs_segment: np.ndarray, token_id: int) -> float:
        """
        GOP = mean log P(correct token | frame) over aligned frames.
        Calibrated to [0, 100] via sigmoid.
        """
        if len(log_probs_segment) == 0:
            return 0.0

        mean_log_prob = float(log_probs_segment[:, token_id].mean())

        # Sigmoid calibration
        score = 100.0 / (1.0 + np.exp(-(self.calibration_alpha * mean_log_prob + self.calibration_beta)))
        return float(np.clip(score, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Alignment + Scoring
    # ------------------------------------------------------------------

    def _align_and_score(
        self,
        log_probs: np.ndarray,
        reference_text: str,
        recognised_text: str,
        frame_alignment: list[AlignmentEntry],
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[WordResult]:
        """
        Score reference words using CTC frame alignment.

        Strategy:
        1. Split the CTC alignment into word groups (separated by | tokens).
        2. Match recognised words to reference words.
        3. For each reference word, use the aligned frames for GOP scoring.
        4. Unmatched reference words get low scores (omission).
        """
        hop_length = 320  # wav2vec2: 16kHz / 50Hz = 320 samples per frame
        ms_per_frame = hop_length / sample_rate * 1000  # ~20ms

        # Group alignment by words (split on word separator |)
        recognised_words = self._split_alignment_by_words(frame_alignment)

        ref_words = reference_text.lower().split()
        rec_words_text = [w["text"] for w in recognised_words]

        # Match reference words to recognised words
        word_results: list[WordResult] = []
        used_rec = set()

        for ref_word in ref_words:
            # Find best matching recognised word
            best_idx = None
            best_match = 0
            for j, rec in enumerate(recognised_words):
                if j in used_rec:
                    continue
                sim = self._word_similarity(ref_word, rec["text"])
                if sim > best_match:
                    best_match = sim
                    best_idx = j

            if best_idx is not None and best_match > 0.3:
                used_rec.add(best_idx)
                rec_word = recognised_words[best_idx]

                # Score phonemes using aligned frames
                phoneme_results = self._score_aligned_phonemes(
                    log_probs, ref_word, rec_word, ms_per_frame
                )

                word_accuracy = (
                    float(np.mean([p.accuracy_score for p in phoneme_results]))
                    if phoneme_results else 0.0
                )

                offset_ms = int(rec_word["start_frame"] * ms_per_frame)
                duration_ms = int(
                    (rec_word["end_frame"] - rec_word["start_frame"]) * ms_per_frame
                )

                error_type = "None" if best_match > 0.7 else "Mispronunciation"

                word_results.append(WordResult(
                    word=ref_word,
                    accuracy_score=word_accuracy,
                    error_type=error_type,
                    offset_ms=offset_ms,
                    duration_ms=duration_ms,
                    phonemes=phoneme_results,
                ))
            else:
                # Reference word not found in recognised output → Omission
                word_results.append(WordResult(
                    word=ref_word,
                    accuracy_score=0.0,
                    error_type="Omission",
                    offset_ms=0,
                    duration_ms=0,
                    phonemes=[],
                ))

        return word_results

    def _split_alignment_by_words(
        self, frame_alignment: list[AlignmentEntry]
    ) -> list[RecognizedWord]:
        """Split CTC alignment into word groups (separated by | tokens)."""
        words: list[RecognizedWord] = []
        current_chars: list[AlignmentEntry] = []
        current_start: int | None = None

        for entry in frame_alignment:
            if entry["char"] == "|":
                # Word boundary
                if current_chars:
                    words.append({
                        "text": "".join(c["char"] for c in current_chars).lower(),
                        "chars": current_chars,
                        "start_frame": current_start if current_start is not None else 0,
                        "end_frame": current_chars[-1]["end_frame"],
                    })
                    current_chars = []
                    current_start = None
            else:
                if current_start is None:
                    current_start = entry["start_frame"]
                current_chars.append(entry)

        # Last word
        if current_chars:
            words.append({
                "text": "".join(c["char"] for c in current_chars).lower(),
                "chars": current_chars,
                "start_frame": current_start if current_start is not None else 0,
                "end_frame": current_chars[-1]["end_frame"],
            })

        return words

    def _score_aligned_phonemes(
        self,
        log_probs: np.ndarray,
        ref_word: str,
        rec_word: RecognizedWord,
        ms_per_frame: float,
    ) -> list[PhonemeResult]:
        """
        Score each character in the reference word using CTC-aligned frames.

        For each ref character, find the matching recognised character's frame
        range and compute GOP there. If no match, use the proportional frame range.
        """
        ref_chars = [c for c in ref_word if c.isalpha()]
        rec_chars = rec_word["chars"]

        if not ref_chars or not rec_chars:
            return []

        results = []

        # Map ref chars to rec chars using simple alignment
        # If recognized "hallo" and ref "hello", map h→h, e→a, l→l, l→l, o→o
        rec_idx = 0
        for ref_char in ref_chars:
            # Find the closest matching rec char
            best_rec = None
            for k in range(rec_idx, len(rec_chars)):
                if rec_chars[k]["char"].lower() == ref_char:
                    best_rec = rec_chars[k]
                    rec_idx = k + 1
                    break

            if best_rec is None and rec_idx < len(rec_chars):
                # No exact match; use next available rec char
                best_rec = rec_chars[min(rec_idx, len(rec_chars) - 1)]
                rec_idx += 1

            if best_rec is not None:
                # Use ONLY the aligned frames — no context expansion
                # Adjacent frames are mostly blank and would dilute the score
                start_f = best_rec["start_frame"]
                end_f = best_rec["end_frame"]

                segment = log_probs[start_f:end_f]

                # Look up token for the REFERENCE character
                token_id = self.processor.tokenizer.convert_tokens_to_ids(ref_char.upper())
                if token_id is None or token_id == self.processor.tokenizer.unk_token_id:
                    accuracy = 50.0
                else:
                    accuracy = self._gop_score(segment, token_id)

                offset_ms = int(start_f * ms_per_frame)
                duration_ms = int((end_f - start_f) * ms_per_frame)
            else:
                # No alignment available
                accuracy = 0.0
                offset_ms = 0
                duration_ms = 0

            results.append(PhonemeResult(
                phoneme=ref_char,
                accuracy_score=accuracy,
                offset_ms=offset_ms,
                duration_ms=duration_ms,
            ))

        return results

    @staticmethod
    def _word_similarity(a: str, b: str) -> float:
        """Simple character-level similarity between two words."""
        if not a or not b:
            return 0.0
        a, b = a.lower(), b.lower()
        if a == b:
            return 1.0
        # Longest common subsequence ratio
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        return 2.0 * lcs / (m + n)
