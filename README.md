# Multimodal Attendance & Engagement Agent

AI agent that estimates attendance and engagement from video (face presence, eye openness,
head pose via OpenCV + MediaPipe) and optional audio (speech activity via Librosa), with a
Streamlit dashboard and teaching recommendations.

## Files
- `app.py` - Streamlit dashboard (run: `streamlit run app.py`)
- `config.py` - all thresholds and weights
- `face_analyzer.py` - vision module
- `audio_analyzer.py` - audio module
- `engagement_agent.py` - score fusion and attendance decision
- `recommender.py` - analytics and recommendations
- `test_core.py` - unit tests

## Scoring
Engagement = 0.6 x attention + 0.4 x audio activity (vision only if no audio).
Present if a face is seen in at least 60% of the session.

## Deploy on Streamlit Cloud
Use Python 3.11 or 3.12 (Advanced settings) because MediaPipe does not support newer versions.
