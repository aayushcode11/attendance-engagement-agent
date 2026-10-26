"""Analytics & recommendation engine: turns the engagement timeline into plain-language advice."""
from __future__ import annotations

from typing import List

import pandas as pd

from config import AGENT


def find_dip_streaks(df: pd.DataFrame, min_len: int = 2) -> List[tuple[float, float]]:
    """Return (start_time, end_time) of consecutive low-engagement windows."""
    streaks, start, prev, count = [], None, None, 0
    for _, row in df.iterrows():
        low = row["engagement"] < AGENT.low_engagement
        if low:
            if start is None:
                start, count = row["t_start"], 0
            count += 1
            prev = row["t_start"]
        else:
            if start is not None and count >= min_len:
                streaks.append((start, prev))
            start = None
    if start is not None and count >= min_len:
        streaks.append((start, prev))
    return streaks


def recommend(df: pd.DataFrame, summary: dict) -> List[str]:
    if df.empty:
        return ["No data captured yet - run an analysis first."]

    tips: List[str] = []
    avg = summary.get("avg_engagement", 0)

    if summary.get("status") == "Absent":
        tips.append("Presence was below the attendance threshold - check camera placement and lighting.")
    if avg >= AGENT.high_engagement:
        tips.append("Great session: engagement stayed high. Keep the same pace and format.")
    elif avg < AGENT.low_engagement:
        tips.append("Overall engagement is low. Try shorter segments, polls or questions to re-involve the participant.")
    else:
        tips.append("Engagement is moderate. Add interaction points every 10-15 minutes.")

    for s, e in find_dip_streaks(df):
        tips.append(f"Engagement dipped between {s:.0f}s and {e:.0f}s - consider a break or topic change around then.")

    if df["audio_active"].notna().any():
        silent = (df["audio_active"].fillna(0) < 0.2).mean()
        if silent > 0.7:
            tips.append("Participant was mostly silent. Invite questions or discussion to encourage speaking.")

    if df["attention"].mean() < 0.5 and df["presence"].mean() > 0.6:
        tips.append("Face is present but attention is low (looking away / eyes closed). Possible distraction or fatigue.")

    return tips
