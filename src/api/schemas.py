"""
Pydantic schemas for the SpeakRight REST API.

The response schema intentionally mirrors Azure's Pronunciation Assessment
JSON output so downstream clients can switch between Azure and SpeakRight
without changing their parsing code.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PronunciationAssessmentRequest(BaseModel):
    reference_text: str = Field(..., description="The word or sentence the speaker intended to say.")
    grading_system: str = Field("HundredMark", description="Scoring scale. Only 'HundredMark' (0–100) supported.")
    granularity: str = Field("Phoneme", description="'Phoneme' | 'Word' | 'FullText'")
    phoneme_alphabet: str = Field("IPA", description="'IPA' only in this version.")
    enable_miscue: bool = Field(False, description="Whether to detect omission / insertion errors.")


# ---------------------------------------------------------------------------
# Response (Azure-compatible)
# ---------------------------------------------------------------------------

class PhonemeScore(BaseModel):
    Phoneme: str
    AccuracyScore: float
    Offset: int   # 100-nanosecond ticks (Azure convention)
    Duration: int


class WordPronunciationAssessment(BaseModel):
    AccuracyScore: float
    ErrorType: str  # "None" | "Omission" | "Insertion" | "Mispronunciation"


class WordScore(BaseModel):
    Word: str
    AccuracyScore: float
    ErrorType: str
    Offset: int
    Duration: int
    Phonemes: list[PhonemeScore]


class NBestEntry(BaseModel):
    Confidence: float
    Lexical: str
    ITN: str
    MaskedITN: str
    Display: str
    PronScore: float
    AccuracyScore: float
    FluencyScore: float
    CompletenessScore: float
    Words: list[WordScore]


class PronunciationAssessmentResponse(BaseModel):
    RecognitionStatus: str
    DisplayText: str
    Offset: int
    Duration: int
    NBest: list[NBestEntry]

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
