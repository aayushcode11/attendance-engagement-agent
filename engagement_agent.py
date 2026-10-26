"""Agent: fuses vision + audio into per-window engagement scores and an attendance decision."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from config import AGENT


@dataclass
class WindowRecord:
    t_start: float
    presence: float          # fraction of frames with a face (0..1)
    attention: float         # fraction of frames attentive (0..1)
    audio_active: Optional[float]  # None when no audio is available
    engagement: float = 0.0
    label: str = ""


def engagement_score(attention: float, audio_active: Optional[float]) -> float:
    """Weighted fusion. Falls back to vision only when audio is missing (audio is optional)."""
    if audio_active is None:
        return float(attention)
    return float(AGENT.weight_attention * attention + AGENT.weight_audio * audio_active)


def label_engagement(score: float, presence: float) -> str:
    if presence < 0.2:
        return "Absent"
    if score >= AGENT.high_engagement:
        return "High"
    if score < AGENT.low_engagement:
        return "Low"
    return "Medium"


@dataclass
class EngagementAgent:
    records: List[WindowRecord] = field(default_factory=list)

    def add_window(
        self,
        t_start: float,
        presence: float,
        attention: float,
        audio_active: Optional[float] = None,
    ) -> WindowRecord:
        rec = WindowRecord(t_start, presence, attention, audio_active)
        rec.engagement = engagement_score(attention * (1 if presence > 0 else 0), audio_active)
        rec.label = label_engagement(rec.engagement, presence)
        self.records.append(rec)
        return rec

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])

    def attendance(self) -> dict:
        if not self.records:
            return {"status": "No data", "presence_pct": 0.0}
        presence = sum(r.presence for r in self.records) / len(self.records)
        status = "Present" if presence >= AGENT.attendance_min_presence else "Absent"
        return {"status": status, "presence_pct": round(presence * 100, 1)}

    def summary(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            return {}
        return {
            **self.attendance(),
            "avg_engagement": round(float(df["engagement"].mean()), 3),
            "avg_attention": round(float(df["attention"].mean()), 3),
            "high_windows": int((df["label"] == "High").sum()),
            "low_windows": int((df["label"].isin(["Low", "Absent"])).sum()),
            "windows": len(df),
        }
