"""
Audio feature extraction utilities.

Primary path: wav2vec2 processes raw waveforms internally — no manual MFCC needed.
MFCCs here are available as a fallback or for classical ML experiments.
"""

import numpy as np
import librosa
import torch
import soundfile as sf
from pathlib import Path


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to target_sr. Returns (waveform, sample_rate)."""
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)  # mono
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    return waveform, target_sr


def extract_mfcc(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,   # 10 ms at 16 kHz
    win_length: int = 400,   # 25 ms
    delta: bool = True,
    delta_delta: bool = True,
) -> np.ndarray:
    """
    Extract MFCC features (optionally with delta and delta-delta).

    Returns array of shape (T, n_features) where n_features = n_mfcc * (1 + delta + delta_delta).
    """
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    features = [mfcc]
    if delta:
        features.append(librosa.feature.delta(mfcc, order=1))
    if delta_delta:
        features.append(librosa.feature.delta(mfcc, order=2))
    return np.concatenate(features, axis=0).T  # (T, C)


def extract_log_mel(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """Log-mel spectrogram. Shape: (T, n_mels)."""
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    return librosa.power_to_db(mel, ref=np.max).T  # (T, n_mels)


def frame_to_time(frame_idx: int, hop_length: int = 160, sr: int = 16000) -> float:
    """Convert frame index to time in seconds."""
    return frame_idx * hop_length / sr


def time_to_sample(time_sec: float, sr: int = 16000) -> int:
    return int(time_sec * sr)
