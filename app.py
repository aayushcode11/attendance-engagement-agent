"""Streamlit dashboard: Multimodal Attendance & Engagement Agent.

Run:  streamlit run app.py
Inputs: webcam snapshots OR an uploaded video file, plus an optional audio file.
Everything runs locally/offline.
"""
from __future__ import annotations

import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from engagement_agent import EngagementAgent
from recommender import recommend
from audio_analyzer import analyze_window, load_audio, split_windows
from config import AGENT, AUDIO

st.set_page_config(page_title="Attendance & Engagement Agent", layout="wide")
st.title("Multimodal Attendance & Engagement Agent")
st.caption("Vision (OpenCV + MediaPipe) + Audio (Librosa) -> attendance, engagement, recommendations")


@st.cache_resource
def get_face_analyzer():
    try:
        from face_analyzer import FaceAnalyzer

        return FaceAnalyzer()
    except Exception as e:  # e.g. MediaPipe not installed / wrong Python version
        st.error(f"Could not start the face analyzer: {e}")
        st.stop()


def process_video(path: str, window_s: int, every_n: int, audio_windows):
    analyzer = get_face_analyzer()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    agent = EngagementAgent()
    frames_per_window = max(int(fps * window_s), 1)

    present = attentive = seen = idx = win = 0
    progress = st.progress(0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    def flush():
        nonlocal present, attentive, seen, win
        if seen == 0:
            return
        audio = float(audio_windows[win].active) if audio_windows and win < len(audio_windows) else None
        agent.add_window(win * window_s, present / seen, attentive / seen, audio)
        present = attentive = seen = 0
        win += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            r = analyzer.analyze(frame)
            present += int(r.present)
            attentive += int(r.attentive)
            seen += 1
        idx += 1
        if idx % frames_per_window == 0:
            flush()
            progress.progress(min(idx / total, 1.0))
    flush()
    cap.release()
    progress.progress(1.0)
    return agent


with st.sidebar:
    st.header("Settings")
    window_s = st.slider("Window size (seconds)", 2, 15, AGENT.window_seconds)
    every_n = st.slider("Analyse every Nth frame", 1, 10, 3, help="Higher = faster, less precise")
    st.markdown("**Weights**: attention %.0f%% / audio %.0f%%" % (AGENT.weight_attention * 100, AGENT.weight_audio * 100))

tab_video, tab_cam = st.tabs(["Video + audio file", "Webcam snapshot"])

with tab_video:
    vid = st.file_uploader("Upload session video", type=["mp4", "avi", "mov", "mkv"])
    aud = st.file_uploader("Upload audio (optional)", type=["wav", "mp3", "flac", "ogg"])
    if vid and st.button("Analyse session", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(vid.read())
            vpath = f.name
        audio_windows = None
        if aud:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(aud.read())
                y = load_audio(f.name)
            audio_windows = [analyze_window(c) for c in split_windows(y, AUDIO.sample_rate, window_s)]
        st.session_state["agent"] = process_video(vpath, window_s, every_n, audio_windows)

with tab_cam:
    shot = st.camera_input("Take a snapshot to check presence & attention")
    if shot:
        arr = np.frombuffer(shot.getvalue(), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        r = get_face_analyzer().analyze(frame)
        c1, c2, c3 = st.columns(3)
        c1.metric("Present", "Yes" if r.present else "No")
        c2.metric("Attentive", "Yes" if r.attentive else "No")
        c3.metric("Yaw / Pitch", f"{r.yaw:.0f} / {r.pitch:.0f}" if r.yaw is not None else "-")

agent = st.session_state.get("agent")
if agent and agent.records:
    df = agent.to_dataframe()
    s = agent.summary()
    st.subheader("Summary")
    c = st.columns(4)
    c[0].metric("Attendance", s["status"], f"{s['presence_pct']}% present")
    c[1].metric("Avg engagement", f"{s['avg_engagement']:.2f}")
    c[2].metric("High windows", s["high_windows"])
    c[3].metric("Low / absent windows", s["low_windows"])

    st.subheader("Engagement timeline")
    st.line_chart(df.set_index("t_start")[["engagement", "attention", "presence"]])

    st.subheader("Recommendations")
    for tip in recommend(df, s):
        st.write("- " + tip)

    with st.expander("Raw data"):
        st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), "engagement_report.csv", "text/csv")
