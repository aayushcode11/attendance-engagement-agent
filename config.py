"""Central configuration (thresholds are easy to tune here)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class VisionConfig:
    ear_closed_threshold: float = 0.20      # eye aspect ratio below => eyes closed
    yaw_limit_deg: float = 25.0             # |yaw| above => looking away sideways
    pitch_limit_deg: float = 20.0           # |pitch| above => looking down/up
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    energy_threshold_db: float = -40.0      # frame louder than this => speech
    min_speech_ratio: float = 0.05          # below => "silent" window


@dataclass(frozen=True)
class AgentConfig:
    attendance_min_presence: float = 0.60   # present >=60% of session => Present
    weight_attention: float = 0.6
    weight_audio: float = 0.4
    window_seconds: int = 5
    high_engagement: float = 0.70
    low_engagement: float = 0.40


VISION = VisionConfig()
AUDIO = AudioConfig()
AGENT = AgentConfig()
