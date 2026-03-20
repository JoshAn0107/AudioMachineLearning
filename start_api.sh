#!/bin/bash
export SPEAKRIGHT_MODEL=facebook/wav2vec2-large-960h
export SPEAKRIGHT_DEVICE=cpu
cd /root/speakright-ml
exec /root/mlenv/bin/uvicorn src.api.server:app --host 0.0.0.0 --port 8002 --workers 1
