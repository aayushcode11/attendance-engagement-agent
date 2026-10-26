import numpy as np
import pandas as pd

from engagement_agent import EngagementAgent, engagement_score, label_engagement
from recommender import find_dip_streaks, recommend
from audio_analyzer import analyze_window
from face_analyzer import classify_attention, eye_aspect_ratio

SR = 16000


def test_attention_rules():
    assert classify_attention(0.3, 0, 0)
    assert not classify_attention(0.1, 0, 0)      # eyes closed
    assert not classify_attention(0.3, 40, 0)     # looking sideways
    assert not classify_attention(0.3, 0, -30)    # looking down


def test_ear_open_vs_closed():
    open_eye = np.array([[0, 0], [1, 1], [2, 1], [3, 0], [2, -1], [1, -1]], float)
    closed = np.array([[0, 0], [1, 0.1], [2, 0.1], [3, 0], [2, -0.1], [1, -0.1]], float)
    assert eye_aspect_ratio(open_eye) > eye_aspect_ratio(closed)


def test_audio_silence_vs_speech():
    silence = np.zeros(SR * 2, dtype=np.float32)
    t = np.linspace(0, 2, SR * 2, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert not analyze_window(silence).active
    assert analyze_window(tone).active


def test_score_fusion_and_fallback():
    assert engagement_score(1.0, 1.0) == 1.0
    assert engagement_score(0.5, None) == 0.5
    assert label_engagement(0.9, 1.0) == "High"
    assert label_engagement(0.1, 1.0) == "Low"
    assert label_engagement(0.9, 0.0) == "Absent"


def test_attendance_and_summary():
    a = EngagementAgent()
    for i in range(10):
        a.add_window(i * 5, presence=1.0 if i < 8 else 0.0, attention=0.9, audio_active=1.0)
    assert a.attendance()["status"] == "Present"
    s = a.summary()
    assert s["windows"] == 10 and s["low_windows"] == 2


def test_dips_and_recommendations():
    a = EngagementAgent()
    for i, att in enumerate([0.9, 0.9, 0.1, 0.1, 0.1, 0.9]):
        a.add_window(i * 5, 1.0, att, None)
    df = a.to_dataframe()
    assert find_dip_streaks(df) == [(10, 20)]
    tips = recommend(df, a.summary())
    assert any("dipped" in t for t in tips)
    assert recommend(pd.DataFrame(), {}) == ["No data captured yet - run an analysis first."]
