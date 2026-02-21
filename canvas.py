"""
canvas.py — Virtual drawing canvas.
Improved: circle-stamp drawing for smoother strokes, proportional eraser.
"""

import cv2
import numpy as np


class Canvas:
    # Maximum pixel distance to bridge across when drawing resumes after a
    # brief finger lift.  Increase to bridge wider gaps; set to 0 to disable.
    MAX_BRIDGE_GAP = 30

    def __init__(self, width: int, height: int):
        self.width         = width
        self.height        = height
        self._canvas       = np.zeros((height, width, 3), dtype=np.uint8)
        self._prev_pt      = None          # last point in the current stroke
        self._last_tip     = None          # fingertip position when drawing last stopped
        self._last_color   = None          # color used in the last stroke
        self._last_thick   = 10            # thickness used in the last stroke

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, x: int, y: int, color: tuple, thickness: int = 10):
        """
        Smooth stroke using circle-stamp interpolation.
        Fills in sub-pixel gaps when the finger moves fast.

        When drawing resumes after a brief lift (pen-up / pen-down), the gap
        between the last drawn point and the new point is bridged automatically
        if the distance is within MAX_BRIDGE_GAP pixels, keeping continuous
        writing connected.
        """
        curr = np.array([x, y], float)

        if self._prev_pt is None:
            # --- Pen-down: check if we should bridge from the last lift point ---
            if (self._last_tip is not None and
                    self.MAX_BRIDGE_GAP > 0 and
                    self._last_color is not None):
                gap = float(np.linalg.norm(curr - self._last_tip))
                if gap <= self.MAX_BRIDGE_GAP:
                    # Bridge the gap with the previous color/thickness so the
                    # stroke looks uninterrupted
                    bridge_steps = max(int(gap / (self._last_thick * 0.4)), 2)
                    for i in range(bridge_steps + 1):
                        t  = i / bridge_steps
                        pt = (self._last_tip + t * (curr - self._last_tip)).astype(int)
                        cv2.circle(self._canvas, tuple(pt),
                                   self._last_thick // 2, color, -1, cv2.LINE_AA)
                    self._prev_pt = curr
                    self._last_tip = None   # consumed
                    self._last_color = color
                    self._last_thick = thickness
                    return

            # No bridge — start fresh with a single dot
            cv2.circle(self._canvas, (x, y), thickness // 2, color, -1,
                       cv2.LINE_AA)
            self._prev_pt = curr
            self._last_color = color
            self._last_thick = thickness
            return

        prev = self._prev_pt
        dist = np.linalg.norm(curr - prev)

        if dist < 1:
            self._prev_pt = curr
            return

        # Smoother & Faster: Use a single line with round caps instead of many circles
        cv2.line(self._canvas, tuple(prev.astype(int)), tuple(curr.astype(int)),
                 color, thickness, cv2.LINE_AA)
        cv2.circle(self._canvas, tuple(prev.astype(int)), thickness // 2,
                   color, -1, cv2.LINE_AA)
        cv2.circle(self._canvas, tuple(curr.astype(int)), thickness // 2,
                   color, -1, cv2.LINE_AA)

        self._prev_pt   = curr
        self._last_color = color
        self._last_thick = thickness

    def erase(self, x: int, y: int, size: int = 55):
        """Erase a circular region (set to black = transparent)."""
        cv2.circle(self._canvas, (x, y), size, (0, 0, 0), -1)
        self._prev_pt = np.array([x, y], float)

    def reset_stroke(self, save_tip: tuple = None):
        """
        Call whenever the DRAW gesture ends so the next draw starts fresh.
        Pass the current fingertip position as `save_tip` so it can be used
        to bridge the gap when drawing resumes.
        """
        if self._prev_pt is not None and save_tip is not None:
            # Only save when we were actually drawing (not just hovering)
            self._last_tip = np.array(save_tip, float)
        self._prev_pt = None

    def clear(self):
        """Wipe entire canvas."""
        self._canvas[:] = 0
        self._prev_pt   = None
        self._last_tip  = None
        self._last_color = None

    # ── Rendering ─────────────────────────────────────────────────────────────

    def blend(self, frame: np.ndarray) -> np.ndarray:
        """Overlay canvas on webcam frame; black = transparent."""
        gray     = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg       = cv2.bitwise_and(frame,         frame,         mask=mask_inv)
        fg       = cv2.bitwise_and(self._canvas,  self._canvas,  mask=mask)
        return cv2.add(bg, fg)
