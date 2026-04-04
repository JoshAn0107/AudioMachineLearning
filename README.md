# SpeakRight ML — Offline Pronunciation Assessment

Offline English pronunciation assessment system designed for coursework benchmarking against Azure Pronunciation Assessment.

## Objective

Build a fully offline scorer that approximates Azure's PronScore pipeline and quantify the gap with reproducible experiments.

- **Target**: Azure-style PronScore, Accuracy, Fluency, Completeness outputs
- **Constraint**: no cloud inference for scoring
- **Evaluation**: side-by-side comparison against Azure on word and sentence audio

## System Architecture

1. **Acoustic model**: `facebook/wav2vec2-large-960h` CTC model
2. **Alignment**: CTC token-frame alignment to map reference characters/phonemes to frame spans
3. **Scoring core**: GOP (Goodness of Pronunciation) on aligned segments
4. **Calibration**: sigmoid mapping from mean log-probability to `[0, 100]`
5. **Aggregation**: word-level scores → utterance-level PronScore components
6. **Serving**: FastAPI endpoints mirroring Azure response schema

## Key Experiments

- **Feature comparison (classical baselines)**
  - MFCC + RF: **81.8%** mean CV accuracy
  - Mel-spectrogram + SVM/RF: **92.7%** mean CV accuracy (best baseline)
  - wav2vec2 pooled embeddings + shallow classifiers: **29.1–38.2%**
- **Calibration grid search**
  - Tuned sigmoid parameters using benchmark alignment objective
  - Best parameters saved in `results/best_calibration.json`: **alpha=3.0, beta=3.5**
- **CTC head fine-tuning**
  - Lightweight head-only fine-tuning experiment (`results/finetuning_report.json`)
  - Char accuracy remained ~**92.9%** in this small-data setting

## Key Results

- **Azure alignment (benchmark.csv)**
  - PronScore **MAE = 2.10** (SpeakRight vs Azure)
  - Indicates close score tracking on the benchmark set
- **Diverse quality tiers (benchmark_diverse.csv)**
  - Mean SpeakRight PronScore: native **96.63**, good **96.26**, medium **85.39**, poor **45.79**, wrong **36.42**, misread **38.96**
  - System separates strong vs degraded pronunciations with clear score drop across poor/wrong tiers
- **Feature comparison takeaway**
  - Classical mel features were strongest on the small baseline task; end-to-end wav2vec2 scoring is retained for Azure-compatible rich outputs

## Reproducibility

### 1) Run API

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Example request:

```bash
curl -X POST http://localhost:8000/pronunciation-assessment/file \
  -F "audio_file=@data/recordings/hello.wav" \
  -F "reference_text=hello"
```

### 2) Run benchmark vs Azure

```bash
python scripts/benchmark_azure.py \
  --words data/test_words.txt \
  --audio-dir data/recordings \
  --output results/benchmark.csv \
  --plot results/comparison_plot.png
```

### 3) Run full evaluation summary

```bash
python scripts/full_evaluation.py
```

Outputs:

- `results/evaluation_summary.json`
- `results/evaluation_summary.png`

### 4) Run tests

```bash
python -m pytest tests/ -v
```

## Project Layout

```text
src/models/wav2vec2_scorer.py    # Offline wav2vec2 + CTC + GOP scorer
scripts/feature_comparison.py    # MFCC/mel/wav2vec2 baseline experiment
scripts/calibrate_gop.py         # Calibration grid search
scripts/finetune_ctc_head.py     # CTC head fine-tuning experiment
scripts/full_evaluation.py       # Consolidated metric + figure generator
results/                         # Experiment outputs and benchmark artifacts
tests/                           # Unit/integration tests with mocks
```

## Known Limitations

- Alignment/scoring is still character-centric in places; phoneme supervision is limited.
- Current benchmark size is modest, so cross-accent robustness is not fully validated.
- Tier boundaries and mispronunciation decision thresholds are heuristic.

## Future Work

- Expand non-native/accented dataset coverage and add stronger held-out protocols.
- Upgrade to phoneme-level forced alignment for tighter error localization.
- Learn calibration and tier thresholds from labeled human ratings instead of fixed heuristics.
