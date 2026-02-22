# SpeakRight — Offline Pronunciation Assessment

An offline English pronunciation scoring system that mirrors the Azure Speech
Pronunciation Assessment API, built on **wav2vec2 + GOP (Goodness of Pronunciation)** scoring.

---

## Project Goal

Compare an offline, custom-built pronunciation scorer against Microsoft Azure's
Pronunciation Assessment API on a set of random English words, then present
findings to the class.

---

## Architecture

```
Audio Input (WAV)
       │
       ▼
Wav2Vec2 (facebook/wav2vec2-base)
  – Encodes raw waveform into contextual representations
  – CTC head → frame-level phoneme log-posteriors
       │
       ▼
Forced Alignment
  – Maps reference phonemes to audio time frames
  – Lightweight: uniform split (default) or Montreal Forced Aligner (recommended)
       │
       ▼
GOP Scoring  (per phoneme)
  – GOP(p) = mean log P(correct phoneme | frames)
  – Calibrated to [0, 100] via sigmoid
       │
       ▼
Score Aggregation
  – AccuracyScore   (0–100)
  – FluencyScore    (0–100)
  – CompletenessScore (0–100)
  – PronScore = 0.4·Acc + 0.2·Flu + 0.4·Com
       │
       ▼
FastAPI REST Endpoint
  – POST /pronunciation-assessment/file
  – POST /pronunciation-assessment/json
  – Response mirrors Azure's JSON schema
```

---

## Setup

```bash
# 1. Create conda/venv environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment config
cp .env.example .env
# Edit .env to add Azure credentials (only needed for benchmark comparison)
```

---

## Running the API Server

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Test it:
```bash
curl -X POST http://localhost:8000/pronunciation-assessment/file \
  -F "audio_file=@my_recording.wav" \
  -F "reference_text=hello"
```

---

## Datasets

| Dataset      | Size        | Phoneme Labels | Best For                        |
|--------------|-------------|----------------|---------------------------------|
| TIMIT        | ~5 h        | Yes            | Phoneme-level supervision       |
| LibriSpeech  | 100–960 h   | No (use MFA)   | Scale, clean native speech      |
| VCTK         | ~44 h       | No (use MFA)   | Multi-speaker accent diversity  |
| L2-ARCTIC    | ~27 h       | Yes            | Non-native pronunciation errors |
| Common Voice | ~2000+ h    | No             | Accent diversity (noisy)        |

Download:
```bash
# LibriSpeech + VCTK (via HuggingFace)
python data/scripts/download_datasets.py --datasets librispeech vctk

# TIMIT (requires LDC licence — download manually first)
python data/scripts/download_datasets.py --datasets timit --timit-dir /path/to/TIMIT

# Preprocess
python data/scripts/preprocess.py --source librispeech --split train_clean_100
```

---

## Training a Custom Model

Fine-tune wav2vec2 on TIMIT phoneme labels for higher accuracy:

```bash
python scripts/train.py \
    --dataset timit \
    --model facebook/wav2vec2-base \
    --output-dir checkpoints/wav2vec2-pron-v1 \
    --epochs 20 \
    --batch-size 16

# Then use the fine-tuned model:
export SPEAKRIGHT_MODEL=checkpoints/wav2vec2-pron-v1
uvicorn src.api.server:app --port 8000
```

---

## Benchmarking vs Azure

```bash
# Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env
# Audio files must be named <word>.wav in the audio directory

python scripts/benchmark_azure.py \
    --words data/test_words.txt \
    --audio-dir data/recordings/ \
    --output results/benchmark.csv \
    --plot results/comparison_plot.png
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
speakright/
├── config/                  Model and dataset configuration
├── data/
│   ├── raw/                 Downloaded datasets (TIMIT, LibriSpeech, VCTK)
│   ├── processed/           Preprocessed HuggingFace datasets
│   └── scripts/             Download and preprocessing scripts
├── src/
│   ├── features/            Audio feature extraction (MFCC, log-mel)
│   ├── models/              wav2vec2 scorer, GOP scorer, base class
│   ├── scoring/             Score aggregation and fluency estimation
│   ├── api/                 FastAPI server, routes, Pydantic schemas
│   └── evaluation/          Metrics, Azure comparison
├── scripts/                 Train, evaluate, and benchmark CLIs
└── tests/                   Unit and integration tests
```

---

## Azure API Compatibility

SpeakRight's response format mirrors Azure's exactly:

```json
{
  "RecognitionStatus": "Success",
  "DisplayText": "Hello",
  "NBest": [{
    "PronScore": 85.2,
    "AccuracyScore": 88.0,
    "FluencyScore": 78.4,
    "CompletenessScore": 100.0,
    "Words": [{
      "Word": "hello",
      "AccuracyScore": 88.0,
      "ErrorType": "None",
      "Phonemes": [
        { "Phoneme": "h", "AccuracyScore": 90.0, "Offset": 0, "Duration": 800000 }
      ]
    }]
  }]
}
```

Sources consulted:
- [Microsoft Learn — Pronunciation Assessment](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-pronunciation-assessment)
- [Witt & Young (2000) — GOP scoring paper](https://doi.org/10.1016/S0167-6393(00)00044-8)
