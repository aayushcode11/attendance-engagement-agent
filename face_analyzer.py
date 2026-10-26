"""Vision module: face presence, eye openness and head pose using MediaPipe + OpenCV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import VISION

# MediaPipe FaceMesh landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
POSE_IDS = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


@dataclass
class FrameResult:
    present: bool
    attentive: bool
    ear: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """pts: 6x2 array ordered [outer, top1, top2, inner, bottom2, bottom1]."""
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return float((v1 + v2) / (2.0 * h + 1e-9))


def classify_attention(ear: float, yaw: float, pitch: float) -> bool:
    """Pure function (easy to unit test): is the person attentive?"""
    eyes_open = ear >= VISION.ear_closed_threshold
    facing = abs(yaw) <= VISION.yaw_limit_deg and abs(pitch) <= VISION.pitch_limit_deg
    return bool(eyes_open and facing)


class FaceAnalyzer:
    def __init__(self) -> None:
        import mediapipe as mp  # imported lazily so tests run without it

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=VISION.min_detection_confidence,
            min_tracking_confidence=VISION.min_tracking_confidence,
        )

    def close(self) -> None:
        self._mesh.close()

    def analyze(self, frame_bgr: np.ndarray) -> FrameResult:
        h, w = frame_bgr.shape[:2]
        res = self._mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return FrameResult(present=False, attentive=False)

        lm = res.multi_face_landmarks[0].landmark
        xy = np.array([(p.x * w, p.y * h) for p in lm])

        ear = (eye_aspect_ratio(xy[LEFT_EYE]) + eye_aspect_ratio(xy[RIGHT_EYE])) / 2
        yaw, pitch = self._head_pose(xy, w, h)
        return FrameResult(True, classify_attention(ear, yaw, pitch), ear, yaw, pitch)

    @staticmethod
    def _head_pose(xy: np.ndarray, w: int, h: int) -> tuple[float, float]:
        image_points = xy[POSE_IDS].astype(np.float64)
        focal = w
        cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            MODEL_POINTS, image_points, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch, yaw = float(angles[0]), float(angles[1])
        # normalise pitch into [-90, 90] (solvePnP often returns values near +-180)
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        return yaw, pitch
