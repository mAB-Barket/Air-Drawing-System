"""
ui_overlay.py — Professional HUD overlay.
New: gesture lock progress bar, eraser cursor hint, cleaner layout.
"""

import cv2
import numpy as np

COLORS = {
    "Red":   (0,   0,   255),
    "Green": (0,   200, 80),
    "Blue":  (255, 100, 0),
    "White": (255, 255, 255),
}

GESTURE_HINTS = [
    "Pinch: Draw",
    "V 2 fingers: Color",
    "Palm: Erase",
    "C: Clear  Q: Quit",
]


class UIOverlay:
    def __init__(self, fw: int, fh: int):
        self.fw   = fw
        self.fh   = fh
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.sm   = cv2.FONT_HERSHEY_SIMPLEX
        self._input_overlay = np.zeros((fh, fw, 3), dtype=np.uint8)
        self._init_static_overlay()

    def _init_static_overlay(self):
        """Pre-renders static UI elements to avoid redrawing them every frame."""
        # Top bar background
        cv2.rectangle(self._input_overlay, (0, 0), (self.fw, 72), (18, 18, 28), -1)
        
        # Bottom hint bar
        h_bar = 28
        y0    = self.fh - h_bar
        cv2.rectangle(self._input_overlay, (0, y0), (self.fw, self.fh), (12, 12, 20), -1)

        # Bottom text
        spc = self.fw // len(GESTURE_HINTS)
        for i, hint in enumerate(GESTURE_HINTS):
            cv2.putText(self._input_overlay, hint, (8 + i * spc, self.fh - 8),
                        self.sm, 0.42, (170, 170, 180), 1, cv2.LINE_AA)
        
        # Divider line
        cv2.line(self._input_overlay, (0, 72), (self.fw, 72), (50, 50, 60), 1)

    def render(self, frame, mode: str, color_name: str,
               color_bgr: tuple, fingers: list, gesture_progress: float = 0.0):
        # Blend the static overlay once
        cv2.addWeighted(self._input_overlay, 0.72, frame, 1.0, 0, frame)

        # Draw dynamic elements on top
        self._top_bar_dynamic(frame, mode, color_name, color_bgr)
        self._progress_bar(frame, gesture_progress, mode)
        self._finger_indicators(frame, fingers)
        return frame

    # ── Sections ──────────────────────────────────────────────────────────────

    def _top_bar_dynamic(self, frame, mode, color_name, color_bgr):
        bar_h = 72
        
        # Mode label with colored accent
        mode_col = {
            "DRAW":   (0,  245, 140),
            "ERASE":  (0,  140, 255),
            "SELECT": (255, 210, 0),
        }.get(mode, (160, 160, 160))

        # Left accent bar
        cv2.rectangle(frame, (0, 0), (5, bar_h), mode_col, -1)

        cv2.putText(frame, f"Mode: {mode}", (18, 46),
                    self.font, 0.85, mode_col, 2, cv2.LINE_AA)

        # Color indicator
        dot_x = self.fw - 190
        cv2.circle(frame, (dot_x, 36), 20, color_bgr, -1)
        cv2.circle(frame, (dot_x, 36), 20, (255, 255, 255), 2)
        cv2.putText(frame, color_name, (dot_x + 30, 44),
                    self.font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    def _progress_bar(self, frame, progress: float, mode: str):
        """Thin bar below the header showing gesture lock progress (0→1)."""
        bar_y  = 72
        bar_h  = 5
        filled = int(self.fw * progress)

        col = {
            "DRAW":   (0,  230, 120),
            "ERASE":  (0,  140, 255),
            "SELECT": (255, 200, 0),
        }.get(mode, (80, 80, 80))

        cv2.rectangle(frame, (0, bar_y), (self.fw, bar_y + bar_h), (35, 35, 45), -1)
        if filled > 0:
            cv2.rectangle(frame, (0, bar_y), (filled, bar_y + bar_h), col, -1)

    def _finger_indicators(self, frame, fingers):
        """5 finger-state dots near the top-right."""
        labels   = ["T", "I", "M", "R", "P"]
        start_x  = self.fw - 185
        y        = 115
        for i, (lbl, state) in enumerate(zip(labels, fingers)):
            cx    = start_x + i * 34
            col   = (0, 220, 110) if state else (55, 55, 65)
            bdr   = (0, 180, 80)  if state else (80, 80, 90)
            cv2.circle(frame, (cx, y), 13, col, -1)
            cv2.circle(frame, (cx, y), 13, bdr,  1)
            cv2.putText(frame, lbl, (cx - 5, y + 5),
                        self.sm, 0.40, (255, 255, 255), 1, cv2.LINE_AA)


