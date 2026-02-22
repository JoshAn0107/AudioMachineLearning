"""Unit tests for the scoring pipeline (no model weights required)."""

import numpy as np
import pytest

from src.models.base_scorer import PhonemeResult, ScoringResult, WordResult
from src.scoring.aggregator import ScoreAggregator
from src.scoring.fluency import estimate_fluency_score
from src.evaluation.metrics import pearson_r, score_mae, error_detection_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_word(word: str, accuracy: float, error: str = "None") -> WordResult:
    return WordResult(
        word=word,
        accuracy_score=accuracy,
        error_type=error,
        offset_ms=0,
        duration_ms=300,
        phonemes=[
            PhonemeResult(phoneme=c, accuracy_score=accuracy, offset_ms=0, duration_ms=100)
            for c in word if c.isalpha()
        ],
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TestScoreAggregator:
    def setup_method(self):
        self.agg = ScoreAggregator()

    def test_perfect_pronunciation(self):
        words = [make_word("hello", 100.0), make_word("world", 100.0)]
        result = self.agg.aggregate(words, "hello world", "hello world", 2000)
        assert result.accuracy_score == pytest.approx(100.0, abs=1.0)
        assert result.completeness_score == pytest.approx(100.0, abs=1.0)
        assert result.recognition_status == "Success"

    def test_missing_word_lowers_completeness(self):
        # Only "hello" recognised, "world" missing
        words = [make_word("hello", 90.0)]
        result = self.agg.aggregate(words, "hello world", "hello", 2000)
        assert result.completeness_score < 100.0

    def test_pron_score_weighted_correctly(self):
        words = [make_word("test", 80.0)]
        result = self.agg.aggregate(words, "test", "test", 1000)
        expected = 0.4 * 80.0 + 0.2 * result.fluency_score + 0.4 * 100.0
        assert result.pron_score == pytest.approx(expected, abs=1.0)

    def test_azure_format_keys(self):
        words = [make_word("hi", 75.0)]
        result = self.agg.aggregate(words, "hi", "hi", 500)
        fmt = result.to_azure_format()
        assert "RecognitionStatus" in fmt
        assert "NBest" in fmt
        assert "PronScore" in fmt["NBest"][0]
        assert "Words" in fmt["NBest"][0]


# ---------------------------------------------------------------------------
# Fluency
# ---------------------------------------------------------------------------

class TestFluency:
    def test_reasonable_speech_rate(self):
        # 3 words in ~1 second → very fast → lower score
        words = [
            WordResult("one", 90.0, "None", 0, 200, []),
            WordResult("two", 90.0, "None", 200, 200, []),
            WordResult("three", 90.0, "None", 400, 200, []),
        ]
        score = estimate_fluency_score(words, 600, "one two three")
        assert 0.0 <= score <= 100.0

    def test_empty_words(self):
        assert estimate_fluency_score([], 1000, "hello") == 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_pearson_perfect(self):
        y = [1.0, 2.0, 3.0, 4.0]
        assert pearson_r(y, y) == pytest.approx(1.0)

    def test_mae_zero_on_perfect(self):
        y = [80.0, 90.0, 70.0]
        assert score_mae(y, y) == pytest.approx(0.0)

    def test_error_detection_all_correct(self):
        true = ["None", "None", "Mispronunciation"]
        pred = ["None", "None", "Mispronunciation"]
        m = error_detection_metrics(true, pred)
        assert m["f1"] == pytest.approx(1.0)

    def test_error_detection_no_detections(self):
        true = ["Mispronunciation", "Mispronunciation"]
        pred = ["None", "None"]
        m = error_detection_metrics(true, pred)
        assert m["recall"] == pytest.approx(0.0)
