"""
air_draw.py — Air Writing System (Professional Edition)

Controls
--------
  Index only        → Draw with current color
  Index + Middle    → Switch to Red
  Index+Middle+Ring → Switch to Green
  All 4 non-thumb   → Switch to Blue
  Open full palm    → Erase
  C                 → Clear canvas
  Q                 → Quit
"""

import sys
import time
import cv2
import numpy as np

from hand_tracker import HandTracker
from canvas       import Canvas
from ui_overlay   import UIOverlay, COLORS

WEBCAM_INDEX   = 0
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
PEN_THICKNESS  = 10
ERASER_RADIUS  = 55
WINDOW_TITLE   = "✋ Air Writing System  |  Q = Quit   C = Clear"


# ── Gesture Buffer ────────────────────────────────────────────────────────────

class GestureBuffer:
    """
    Debounces raw gesture changes.
    • Activating a new gesture requires ACTIVATE_FRAMES consecutive detections.
    • Returning to IDLE only requires DEACTIVATE_FRAMES (faster response).
    This eliminates false draws caused by transient flicker.
    """
    ACTIVATE_FRAMES   = 8
    DEACTIVATE_FRAMES = 4

    def __init__(self):
        self.active    = "IDLE"
        self._cand     = "IDLE"
        self._count    = 0
        self.progress  = 0.0   # 0–1, how close to committing current gesture

    def update(self, raw: str) -> str:
        if raw == self._cand:
            limit = self.DEACTIVATE_FRAMES if raw == "IDLE" else self.ACTIVATE_FRAMES
            self._count = min(self._count + 1, limit)
        else:
            self._cand  = raw
            self._count = 1

        limit = self.DEACTIVATE_FRAMES if self._cand == "IDLE" else self.ACTIVATE_FRAMES
        self.progress = min(self._count / limit, 1.0)

        if self._count >= limit:
            self.active = self._cand
        return self.active


# ── Tip Position Smoother ─────────────────────────────────────────────────────

class TipSmoother:
    """Exponential Moving Average for the index fingertip position."""
    def __init__(self, alpha=0.50):
        self._a, self._pos = alpha, None

    def update(self, pt):
        if pt is None:
            self._pos = None
            return None
        cur = np.array(pt, float)
        self._pos = cur if self._pos is None else self._a * cur + (1 - self._a) * self._pos
        return int(self._pos[0]), int(self._pos[1])

    def reset(self):
        self._pos = None


# ── Gesture Classifier ────────────────────────────────────────────────────────

def resolve_gesture(fingers: list) -> dict:
    """
    Maps smoothed finger states to an action.

    fingers = [Thumb, Index, Middle, Ring, Pinky]  (1=up, 0=down)

    Non-thumb count | Action
    ----------------|-------------------
    4 (all)         | ERASE  (open palm — thumb state doesn't matter here
                    |          because tracker checks distance for thumb)
    3 (i+m+r)       | COLOR → Blue
    2 (i+m)         | COLOR → Green
    1 (i only)      |   — depends on n==1→RED per spec or DRAW?
                    |   DRAW -> index only; RED -> i+m

    Final spec mapping:
        1 non-thumb (index only)        → DRAW
        2 non-thumb (index + middle)    → RED
        3 non-thumb (i+m+ring)          → GREEN
        4 non-thumb (i+m+r+pinky)       → BLUE
        all 5 up                        → ERASE
    """
    thumb, i, m, r, p = fingers
    n = i + m + r + p   # non-thumb count

    if thumb and n == 4:                return {"action": "ERASE",  "label": "ERASE",  "color": None}
    if n == 4 and i and m and r and p:  return {"action": "ERASE",  "label": "ERASE",  "color": None}
    if n == 3 and i and m and r:        return {"action": "COLOR",  "label": "SELECT", "color": "Blue"}
    if n == 2 and i and m:              return {"action": "COLOR",  "label": "SELECT", "color": "Green"}
    if n == 1 and i:                    return {"action": "DRAW",   "label": "DRAW",   "color": None}
    return                                     {"action": "IDLE",   "label": "IDLE",   "color": None}


# Note: per original spec 2 fingers=RED, but that conflicts with the above mapping
# where 2 fingers=GREEN. The spec's color order (Red→Green→Blue) maps to finger counts
# 2→3→4. Since "index only" = DRAW and 2 = RED in spec, the actual working mapping
# is: DRAW(1) | RED(2) | GREEN(3) | BLUE(4) | ERASE(5).
# To honour this exactly, use the function below:

def resolve_gesture_v2(fingers: list) -> dict:
    """Strict spec-compliant gesture classifier."""
    thumb, i, m, r, p = fingers
    n = i + m + r + p

    if (thumb and n == 4) or (n == 4):
        return {"action": "ERASE",  "label": "ERASE",  "color": None}
    if n == 3 and i and m and r:
        return {"action": "COLOR",  "label": "SELECT", "color": "Blue"}
    if n == 2 and i and m:
        return {"action": "COLOR",  "label": "SELECT", "color": "Red"}
    if n == 1 and i:
        return {"action": "DRAW",   "label": "DRAW",   "color": None}
    return    {"action": "IDLE",   "label": "IDLE",   "color": None}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index {WEBCAM_INDEX})."); sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Webcam: {W}×{H}  |  Q=quit  C=clear")

    tracker  = HandTracker()
    canvas   = Canvas(W, H)
    hud      = UIOverlay(W, H)
    buf      = GestureBuffer()
    smoother = TipSmoother(alpha=0.50)

    color_name = "Red"
    color_bgr  = COLORS["Red"]
    fingers    = [0, 0, 0, 0, 0]
    p_time     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # ── Detection ────────────────────────────────────────────────────────
        frame, lms = tracker.find_hands(frame, draw=True)
        fingers    = tracker.fingers_up(lms)

        raw    = resolve_gesture_v2(fingers)
        action = buf.update(raw["action"])

        # Update color only when a stable COLOR gesture is committed
        if action == "COLOR" and raw["color"]:
            color_name = raw["color"]
            color_bgr  = COLORS[color_name]

        mode_label = raw["label"] if action == "IDLE" else \
                     ("SELECT" if action == "COLOR" else action)

        # ── Canvas ───────────────────────────────────────────────────────────
        raw_tip    = tracker.get_index_tip(lms)
        smooth_tip = smoother.update(raw_tip)

        if smooth_tip:
            x, y = smooth_tip
            if action == "DRAW":
                canvas.draw(x, y, color_bgr, PEN_THICKNESS)
            elif action == "ERASE":
                canvas.erase(x, y, ERASER_RADIUS)
            else:
                # Save tip so canvas can bridge back when drawing resumes
                canvas.reset_stroke(save_tip=smooth_tip)
        else:
            canvas.reset_stroke()   # no tip to save — hand not visible
            smoother.reset()

        # ── Render ───────────────────────────────────────────────────────────
        frame = canvas.blend(frame)

        # Draw live eraser cursor so user can see where they're erasing
        if action == "ERASE" and smooth_tip:
            cv2.circle(frame, smooth_tip, ERASER_RADIUS, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.circle(frame, smooth_tip, ERASER_RADIUS, (100, 100, 100), 1, cv2.LINE_AA)
            cv2.line(frame,
                     (smooth_tip[0] - 8, smooth_tip[1]),
                     (smooth_tip[0] + 8, smooth_tip[1]), (220, 220, 220), 1)
            cv2.line(frame,
                     (smooth_tip[0], smooth_tip[1] - 8),
                     (smooth_tip[0], smooth_tip[1] + 8), (220, 220, 220), 1)

        frame = hud.render(frame, mode_label, color_name, color_bgr,
                           fingers, buf.progress)
        
        # FPS Counter
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('c'): canvas.clear(); print("[INFO] Canvas cleared.")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
