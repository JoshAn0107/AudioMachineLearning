#!/usr/bin/env python3
"""
Generate diverse test audio with varying quality levels for benchmarking.

Creates audio at 5 quality tiers:
  - native:  Clean gTTS (baseline, ~90-100 scores expected)
  - good:    Slight noise + minor speed variation (~75-90)
  - medium:  Moderate noise + pitch shift (~55-75)
  - poor:    Heavy noise + wrong speed + clipping (~30-55)
  - wrong:   Wrong word entirely or severe distortion (~0-30)
"""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import speedup
from gtts import gTTS

AUDIO_DIR = Path("data/diverse_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Words to test at each quality level
TEST_WORDS = [
    "hello", "beautiful", "technology", "apple", "environment",
    "congratulations", "university", "responsibility"
]

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "She sells seashells by the seashore",
    "I would like a cup of coffee please",
]


def tts_to_wav(text: str, output_path: str, slow: bool = False) -> bool:
    """Generate speech via gTTS and convert to 16kHz WAV."""
    try:
        tts = gTTS(text=text, lang="en", slow=slow)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = tmp.name
            tts.save(tmp_mp3)
        audio = AudioSegment.from_mp3(tmp_mp3)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        os.remove(tmp_mp3)
        return True
    except Exception as e:
        print(f"  TTS failed: {e}")
        return False


def add_noise(audio_arr: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at specified SNR."""
    signal_power = np.mean(audio_arr ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio_arr)).astype(np.float32) * np.sqrt(noise_power)
    return audio_arr + noise


def change_pitch(audio_arr: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by resampling."""
    factor = 2 ** (semitones / 12.0)
    new_len = int(len(audio_arr) / factor)
    indices = np.linspace(0, len(audio_arr) - 1, new_len).astype(int)
    return audio_arr[indices]


def change_speed(audio_arr: np.ndarray, factor: float) -> np.ndarray:
    """Change speed by resampling (1.5 = 50% faster)."""
    new_len = int(len(audio_arr) / factor)
    indices = np.linspace(0, len(audio_arr) - 1, new_len).astype(int)
    indices = np.clip(indices, 0, len(audio_arr) - 1)
    return audio_arr[indices]


def add_reverb(audio_arr: np.ndarray, sr: int, decay: float = 0.3) -> np.ndarray:
    """Simple reverb simulation via delayed echo."""
    delay_samples = int(sr * 0.05)  # 50ms delay
    reverb = np.zeros_like(audio_arr)
    reverb[delay_samples:] = audio_arr[:-delay_samples] * decay
    return audio_arr + reverb


def clip_audio(audio_arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Simulate clipping/distortion."""
    return np.clip(audio_arr, -threshold, threshold)


def truncate_word(audio_arr: np.ndarray, keep_ratio: float = 0.6) -> np.ndarray:
    """Truncate audio to simulate incomplete pronunciation."""
    keep_len = int(len(audio_arr) * keep_ratio)
    result = np.zeros_like(audio_arr)
    result[:keep_len] = audio_arr[:keep_len]
    # Fade out
    fade_len = min(1600, keep_len // 4)  # 100ms fade
    fade = np.linspace(1.0, 0.0, fade_len)
    result[keep_len - fade_len:keep_len] *= fade
    return result


def generate_quality_variant(
    base_wav: str, output_path: str, quality: str, sr: int = 16000
) -> bool:
    """Generate a quality variant of the base audio."""
    audio, file_sr = sf.read(base_wav, dtype="float32")
    if file_sr != sr:
        # Simple resample
        ratio = sr / file_sr
        audio = change_speed(audio, 1.0 / ratio)

    if quality == "native":
        # Clean — just copy
        sf.write(output_path, audio, sr)

    elif quality == "good":
        # Slight noise (30dB SNR) + minor speed variation
        audio = add_noise(audio, snr_db=30)
        speed_factor = random.choice([0.92, 0.95, 1.05, 1.08])
        audio = change_speed(audio, speed_factor)
        sf.write(output_path, audio, sr)

    elif quality == "medium":
        # Moderate noise (18dB) + pitch shift + slight reverb
        audio = add_noise(audio, snr_db=18)
        semitones = random.choice([-3, -2, 2, 3])
        audio = change_pitch(audio, sr, semitones)
        audio = add_reverb(audio, sr, decay=0.2)
        sf.write(output_path, audio, sr)

    elif quality == "poor":
        # Heavy noise (10dB) + wrong speed + clipping
        audio = add_noise(audio, snr_db=10)
        audio = change_speed(audio, random.choice([0.7, 1.4]))
        audio = clip_audio(audio, threshold=0.4)
        audio = add_reverb(audio, sr, decay=0.4)
        sf.write(output_path, audio, sr)

    elif quality == "wrong":
        # Severe distortion OR truncated
        choice = random.choice(["truncate", "noise_only", "reversed"])
        if choice == "truncate":
            audio = truncate_word(audio, keep_ratio=random.uniform(0.3, 0.5))
            audio = add_noise(audio, snr_db=15)
        elif choice == "noise_only":
            # Mostly noise with faint speech
            audio = add_noise(audio, snr_db=3)
            audio = clip_audio(audio, threshold=0.3)
        elif choice == "reversed":
            # Reverse the audio (sounds nothing like the word)
            audio = audio[::-1].copy()
            audio = add_noise(audio, snr_db=20)
        sf.write(output_path, audio, sr)

    return True


def main():
    random.seed(42)
    np.random.seed(42)

    qualities = ["native", "good", "medium", "poor", "wrong"]
    all_texts = TEST_WORDS + TEST_SENTENCES

    total = 0
    for text in all_texts:
        safe_name = text.lower().replace(" ", "_")[:40]

        # Generate base TTS audio first
        base_path = AUDIO_DIR / f"{safe_name}_base.wav"
        if not base_path.exists():
            print(f"Generating TTS: {text}")
            if not tts_to_wav(text, str(base_path)):
                continue
            import time
            time.sleep(0.5)
        else:
            print(f"Using cached TTS: {text}")

        # Generate each quality variant
        for quality in qualities:
            out_path = AUDIO_DIR / f"{safe_name}_{quality}.wav"
            if not out_path.exists():
                generate_quality_variant(str(base_path), str(out_path), quality)
                print(f"  [{quality:8s}] {out_path.name}")
                total += 1
            else:
                print(f"  [{quality:8s}] (cached) {out_path.name}")
                total += 1

    # Also generate "wrong word" variants — say a different word
    wrong_word_pairs = [
        ("hello", "yellow"),
        ("apple", "maple"),
        ("university", "anniversary"),
        ("beautiful", "dutiful"),
    ]
    for ref_word, wrong_word in wrong_word_pairs:
        safe = f"{ref_word}_misread"
        out_path = AUDIO_DIR / f"{safe}.wav"
        if not out_path.exists():
            print(f"Generating misread: '{wrong_word}' for reference '{ref_word}'")
            tts_to_wav(wrong_word, str(out_path))
            total += 1
            import time
            time.sleep(0.5)

    print(f"\nGenerated {total} audio files in {AUDIO_DIR}/")
    print(f"Total files: {len(list(AUDIO_DIR.glob('*.wav')))}")


if __name__ == "__main__":
    main()
