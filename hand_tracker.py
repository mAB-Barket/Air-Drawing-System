"""
hand_tracker.py
Improved: adaptive margin-based finger detection + temporal smoothing + better thumb logic.
Compatible with MediaPipe 0.10.x+ (Tasks API).
"""

import os, sys, math, urllib.request
from collections import deque
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

FINGER_TIPS          = [4, 8, 12, 16, 20]
FINGER_PIPS_NOTHUMB  = [6, 10, 14, 18]    # PIP joints for index→pinky
WRIST, MID_MCP       = 0, 9               # landmarks for hand-size reference

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,5),(0,17),
]

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading MediaPipe Hand Landmarker model (~8 MB)…")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[INFO] Download complete.")
        except Exception as e:
            print(f"[ERROR] Download failed: {e}"); sys.exit(1)
    return MODEL_PATH


class HandTracker:
    """
    MediaPipe HandLandmarker wrapper with:
      • Adaptive margin: finger 'up' only when tip is >N% of hand-size above its PIP.
      • Distance-based thumb detection (reliable for both hands / mirror mode).
      • Temporal smoothing: majority vote over a rolling history window.
    """

    def __init__(self, max_hands=1, det_conf=0.72, track_conf=0.65,
                 history_len=6, margin_ratio=0.13):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
            num_hands=max_hands,
        )
        self._lm          = mp_vision.HandLandmarker.create_from_options(opts)
        self._ts          = 0
        self._history     = deque(maxlen=history_len)
        self.margin_ratio = margin_ratio

    # ── Public API ───────────────────────────────────────────────────────────

    def find_hands(self, frame, draw=True):
        h, w = frame.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._ts += 33
        res, lms, pts = self._lm.detect_for_video(mp_img, self._ts), [], []

        if res.hand_landmarks:
            for i, lm in enumerate(res.hand_landmarks[0]):
                px, py = int(lm.x * w), int(lm.y * h)
                lms.append({'id': i, 'x': px, 'y': py, 'z': lm.z})
                pts.append((px, py))
            if draw:
                self._draw_skeleton(frame, pts)

        return frame, lms

    def fingers_up(self, lms):
        """
        Returns [Thumb, Index, Middle, Ring, Pinky] — 1=UP, 0=DOWN.

        Finger is UP only if its tip is at least `margin_ratio * hand_size`
        pixels above its PIP joint.  Smoothed via majority-vote history.
        """
        if len(lms) < 21:
            self._history.clear()
            return [0, 0, 0, 0, 0]

        # Adaptive threshold from hand size
        wrist   = np.array([lms[WRIST]['x'],   lms[WRIST]['y']],   float)
        mid_mcp = np.array([lms[MID_MCP]['x'], lms[MID_MCP]['y']], float)
        hand_sz = float(np.linalg.norm(wrist - mid_mcp))
        margin  = hand_sz * self.margin_ratio

        f = [0] * 5

        # Thumb — distance from tip (4) to index MCP (5); reliable in mirror mode
        thumb_tip  = np.array([lms[4]['x'], lms[4]['y']], float)
        index_mcp  = np.array([lms[5]['x'], lms[5]['y']], float)
        f[0] = 1 if np.linalg.norm(thumb_tip - index_mcp) > hand_sz * 0.55 else 0

        # Index→Pinky: tip.y must be clearly above PIP.y
        for fi, (tip_i, pip_i) in enumerate(
                zip(FINGER_TIPS[1:], FINGER_PIPS_NOTHUMB), 1):
            f[fi] = 1 if lms[tip_i]['y'] < lms[pip_i]['y'] - margin else 0

        # Majority-vote smoothing over history
        self._history.append(f)
        n = len(self._history)
        return [1 if sum(h[i] for h in self._history) > n // 2 else 0
                for i in range(5)]

    def get_index_tip(self, lms):
        """Return (x, y) of index fingertip, or None."""
        return (lms[8]['x'], lms[8]['y']) if len(lms) >= 9 else None

    def close(self):
        self._lm.close()

    # ── Private ──────────────────────────────────────────────────────────────

    def _draw_skeleton(self, frame, pts):
        # Draw connections
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (50, 190, 100), 2, cv2.LINE_AA)
        
        # Draw landmarks - vectorized-style loop for speed
        for i, pt in enumerate(pts):
            if i in FINGER_TIPS:
                cv2.circle(frame, pt, 7, (0, 255, 160), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, (10, 10, 10), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, (220, 220, 220), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, (10, 10, 10), 1, cv2.LINE_AA)
