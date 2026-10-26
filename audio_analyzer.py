"""Audio module: energy-based voice activity detection + simple spectral features (librosa)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config import AUDIO


@dataclass
class AudioWindow:
    speech_ratio: float   # fraction of frames containing speech (0..1)
    mean_db: float        # average loudness in dB
    active: bool          # True if the person is audibly participating


def frame_signal(y: np.ndarray, sr: int, frame_ms: int = AUDIO.frame_ms) -> np.ndarray:
    n = int(sr * frame_ms / 1000)
    if n <= 0 or len(y) < n:
        return np.empty((0, max(n, 1)))
    usable = (len(y) // n) * n
    return y[:usable].reshape(-1, n)


def frame_db(frames: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def analyze_window(y: np.ndarray, sr: int = AUDIO.sample_rate) -> AudioWindow:
    """Analyse one chunk of mono float audio in [-1, 1]."""
    frames = frame_signal(np.asarray(y, dtype=np.float32), sr)
    if len(frames) == 0:
        return AudioWindow(0.0, -120.0, False)
    db = frame_db(frames)
    speech = db > AUDIO.energy_threshold_db
    ratio = float(np.mean(speech))
    return AudioWindow(ratio, float(np.mean(db)), ratio >= AUDIO.min_speech_ratio)


def load_audio(path: str, sr: int = AUDIO.sample_rate) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def split_windows(y: np.ndarray, sr: int, window_seconds: float) -> List[np.ndarray]:
    n = int(sr * window_seconds)
    return [y[i : i + n] for i in range(0, len(y), n) if len(y[i : i + n]) > 0]


def spectral_summary(y: np.ndarray, sr: int = AUDIO.sample_rate) -> dict:
    """Optional extra features (centroid, zero-crossing) for the dashboard."""
    import librosa

    return {
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
    }
