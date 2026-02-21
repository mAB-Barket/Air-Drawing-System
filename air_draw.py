"""
air_draw.py  —  Air Writing System  (v3 — State-Machine Architecture)

All gesture smoothing is handled by ONE component (GestureStateMachine).
No conflicting smoothing layers.  Sticky states, clean transitions.

Controls
--------
  Pinch (thumb+index)  ->  Draw
  Unpinch              ->  Stop drawing
  Index + Middle (V)   ->  Cycle color
  Full open palm       ->  Erase
  C                    ->  Clear canvas
  Q                    ->  Quit
"""

import sys
import ctypes
import cv2
import numpy as np
from collections import deque

from hand_tracker import HandTracker
from canvas       import Canvas
from ui_overlay   import UIOverlay, COLORS

WEBCAM_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
PEN_THICKNESS = 5
ERASER_RADIUS = 60
WINDOW_TITLE  = "Air Writing System"

COLOR_CYCLE = ["Blue", "Red", "Green", "White"]

# ─────────────────────────────────────────────────────────────────────────────
#  Cursor Smoother  —  Double EMA for very stable drawing position
# ─────────────────────────────────────────────────────────────────────────────

class CursorSmoother:
    """
    Single EMA with high alpha (0.7) = very responsive, minimal lag.
    Just enough smoothing to remove pixel-level jitter.
    """
    def __init__(self, alpha=0.70):
        self._a = alpha
        self._pos = None

    def update(self, pt):
        if pt is None:
            return None
        cur = np.array(pt, float)
        if self._pos is None:
            self._pos = cur.copy()
        else:
            self._pos = self._a * cur + (1 - self._a) * self._pos
        return int(self._pos[0]), int(self._pos[1])

    def reset(self):
        self._pos = None


# ─────────────────────────────────────────────────────────────────────────────
#  Gesture State Machine  —  Single source of truth for ALL gesture logic
# ─────────────────────────────────────────────────────────────────────────────

class GestureStateMachine:
    """
    States: IDLE, DRAW, COLOR, ERASE

    Transition rules (tuned for ~30 fps, minimal lag):

    IDLE -> DRAW:   pinch ratio < 0.28  INSTANT (1 frame)
    DRAW -> IDLE:   pinch ratio > 0.40 for 6 consecutive frames (sticky)
    DRAW -> ERASE:  BLOCKED  (must go through IDLE first)
    DRAW -> COLOR:  BLOCKED  (must go through IDLE first)

    IDLE -> ERASE:  all 5 fingers up for 10 consecutive frames
    IDLE -> COLOR:  index+middle up, ring+pinky down for 4 frames

    ERASE -> IDLE:  condition not met for 3 frames
    COLOR -> IDLE:  condition not met for 3 frames

    DRAW is instant-on and sticky — starts immediately, requires clear unpinch to stop.
    """

    def __init__(self):
        self.state = "IDLE"
        self._counter = 0          # frames the candidate has been seen
        self._candidate = "IDLE"   # what we're counting toward
        self.progress = 0.0

        # Rolling history for finger state (majority vote) — small buffer = fast
        self._finger_buf = deque(maxlen=5)

    def _smooth_fingers(self, raw_fingers):
        """Majority-vote over last 5 frames — fast response, still filters noise."""
        self._finger_buf.append(raw_fingers)
        n = len(self._finger_buf)
        return [1 if sum(h[i] for h in self._finger_buf) > n // 2 else 0
                for i in range(5)]

    def update(self, raw_fingers, pinch_ratio):
        """
        pinch_ratio = pinch_distance / hand_size  (0 = fully pinched, >1 = wide open)
        Returns current state string.
        """
        fingers = self._smooth_fingers(raw_fingers)
        thumb, idx, mid, ring, pinky = fingers
        non_thumb = idx + mid + ring + pinky

        # Determine what the raw input is suggesting right now
        raw_suggestion = self._classify_raw(fingers, non_thumb, thumb, idx, mid, ring, pinky, pinch_ratio)

        # State-specific transition logic
        if self.state == "IDLE":
            # DRAW is INSTANT — no waiting
            if raw_suggestion == "DRAW":
                self.state = "DRAW"
                self._counter = 0
                self._candidate = "DRAW"
                self.progress = 1.0
            else:
                self._try_transition(raw_suggestion, {
                    "ERASE": 10,
                    "COLOR": 4,
                })

        elif self.state == "DRAW":
            # From DRAW, can ONLY go to IDLE — requires brief unpinch
            if raw_suggestion != "DRAW":
                raw_suggestion = "IDLE"
            self._try_transition(raw_suggestion, {
                "IDLE": 3,   # 3 frames of unpinch to stop — fast enough to lift between letters
            })

        elif self.state == "ERASE":
            if raw_suggestion != "ERASE":
                raw_suggestion = "IDLE"
            self._try_transition(raw_suggestion, {
                "IDLE": 3,
            })

        elif self.state == "COLOR":
            if raw_suggestion != "COLOR":
                raw_suggestion = "IDLE"
            self._try_transition(raw_suggestion, {
                "IDLE": 3,
            })

        return self.state

    def _classify_raw(self, fingers, non_thumb, thumb, idx, mid, ring, pinky, pinch_ratio):
        """Pure classification — no state awareness."""
        # ERASE: full open hand (all 5 fingers clearly up)
        if thumb and non_thumb == 4 and idx and mid and ring and pinky:
            return "ERASE"
        # COLOR: V-sign (index + middle only, others down)
        if idx and mid and not ring and not pinky:
            return "COLOR"
        # DRAW: pinch (thumb and index close together)
        if pinch_ratio >= 0 and pinch_ratio < 0.28:
            return "DRAW"
        return "IDLE"

    def _try_transition(self, suggestion, thresholds):
        """Count consecutive frames of a suggestion, transition if threshold met."""
        if suggestion == self.state:
            # Already in this state — reset counter
            self._candidate = self.state
            self._counter = 0
            self.progress = 1.0
            return

        if suggestion == self._candidate:
            self._counter += 1
        else:
            self._candidate = suggestion
            self._counter = 1

        needed = thresholds.get(suggestion, 8)
        self.progress = min(self._counter / needed, 1.0)

        if self._counter >= needed:
            self.state = suggestion
            self._counter = 0
            self._candidate = suggestion

    @property
    def fingers(self):
        """Last smoothed finger state (for UI display)."""
        if self._finger_buf:
            n = len(self._finger_buf)
            return [1 if sum(h[i] for h in self._finger_buf) > n // 2 else 0
                    for i in range(5)]
        return [0, 0, 0, 0, 0]


# ─────────────────────────────────────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index {WEBCAM_INDEX})."); sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Webcam: {W}x{H}  |  Q=quit  C=clear")

    # Window: 1/4 of screen, centered
    user32   = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    win_w, win_h = screen_w // 2, screen_h // 2
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, win_w, win_h)
    cv2.moveWindow(WINDOW_TITLE, screen_w // 4, screen_h // 4)

    tracker  = HandTracker()
    canvas   = Canvas(W, H)
    hud      = UIOverlay(W, H)
    sm       = GestureStateMachine()
    cursor   = CursorSmoother(alpha=0.70)

    color_name = "Blue"
    color_bgr  = COLORS["Blue"]
    color_idx  = 0
    was_color  = False   # debounce for color cycling

    while True:
        # Window-close detection
        try:
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
                break
        except cv2.error:
            break

        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # ── Raw detection ────────────────────────────────────────────────────
        frame, lms = tracker.find_hands(frame, draw=True)
        raw_fingers = tracker.raw_fingers_up(lms)
        hand_sz     = tracker.hand_size(lms)
        pinch_dist  = tracker.pinch_distance(lms)

        # Normalized pinch ratio (0 = closed, 1+ = wide open).  -1 = no hand
        pinch_ratio = pinch_dist / hand_sz if hand_sz > 0 else -1.0

        # ── State machine ────────────────────────────────────────────────────
        action = sm.update(raw_fingers, pinch_ratio)
        fingers = sm.fingers   # smoothed, for UI display

        # Color cycling (once per gesture entry)
        if action == "COLOR":
            if not was_color:
                color_idx  = (color_idx + 1) % len(COLOR_CYCLE)
                color_name = COLOR_CYCLE[color_idx]
                color_bgr  = COLORS[color_name]
            was_color = True
        else:
            was_color = False

        # ── Cursor position ──────────────────────────────────────────────────
        # Use pinch midpoint when drawing (more stable), index tip otherwise
        if action == "DRAW":
            raw_pt = tracker.pinch_midpoint(lms)
        else:
            raw_pt = tracker.get_index_tip(lms)

        smooth_pt = cursor.update(raw_pt)

        # ── Canvas actions ───────────────────────────────────────────────────
        if smooth_pt:
            x, y = smooth_pt
            if action == "DRAW":
                canvas.draw(x, y, color_bgr, PEN_THICKNESS)
            elif action == "ERASE":
                canvas.erase(x, y, ERASER_RADIUS)
            else:
                canvas.reset_stroke(save_tip=smooth_pt)
        else:
            canvas.reset_stroke()
            cursor.reset()

        # ── Render ───────────────────────────────────────────────────────────
        frame = canvas.blend(frame)

        # Eraser cursor
        if action == "ERASE" and smooth_pt:
            cv2.circle(frame, smooth_pt, ERASER_RADIUS, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.circle(frame, smooth_pt, ERASER_RADIUS, (100, 100, 100), 1, cv2.LINE_AA)
            cv2.line(frame,
                     (smooth_pt[0] - 8, smooth_pt[1]),
                     (smooth_pt[0] + 8, smooth_pt[1]), (220, 220, 220), 1)
            cv2.line(frame,
                     (smooth_pt[0], smooth_pt[1] - 8),
                     (smooth_pt[0], smooth_pt[1] + 8), (220, 220, 220), 1)

        # Draw-cursor dot (visual feedback when pinching)
        if action == "DRAW" and smooth_pt:
            cv2.circle(frame, smooth_pt, 6, color_bgr, -1, cv2.LINE_AA)
            cv2.circle(frame, smooth_pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

        mode_label = "SELECT" if action == "COLOR" else action
        frame = hud.render(frame, mode_label, color_name, color_bgr,
                           fingers, sm.progress)

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('c'): canvas.clear(); print("[INFO] Canvas cleared.")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
