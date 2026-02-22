# Air Writing System

Write in thin air using just your hand. A real-time gesture-controlled drawing application built with Python, OpenCV, and MediaPipe.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## How It Works

Your webcam tracks 21 hand landmarks in real-time. A custom **gesture state machine** interprets your hand pose and translates it into drawing actions — no physical contact with any surface required.

---

## Gesture Controls

| Gesture | Action |
|---------|--------|
| **Pinch** (thumb + index finger together) | Start drawing |
| **Release pinch** | Stop drawing — reposition freely |
| **V-sign** (index + middle finger up) | Cycle pen color |
| **Open palm** (all fingers spread) | Erase mode |
| `C` key | Clear entire canvas |
| `Q` key / Close window | Quit |

**Available colors:** Blue (default) → Red → Green → White → Blue ...

---

## Quick Start

### Prerequisites

- Python 3.8+
- A webcam

### Installation

```bash
git clone https://github.com/mAB-Barket/Air-Drawing-System.git
cd Air-Drawing-System
pip install -r requirements.txt
```

### Run

```bash
python air_draw.py
```

The window opens at 1/4 of your screen size, centered. Start pinching to write.

---

## Project Structure

```
Air-Drawing-System/
├── air_draw.py         # Main app — state machine, main loop, cursor smoothing
├── hand_tracker.py     # MediaPipe wrapper — raw hand landmark detection
├── canvas.py           # Virtual drawing canvas with stroke management
├── ui_overlay.py       # HUD overlay — mode display, color indicator, hints
├── hand_landmarker.task# MediaPipe model file (auto-downloaded on first run)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Architecture

```
Webcam Frame
     │
     ▼
 HandTracker          ← Raw landmarks, pinch distance, finger states
     │
     ▼
 GestureStateMachine  ← Single source of truth: IDLE / DRAW / COLOR / ERASE
     │
     ▼
 CursorSmoother       ← EMA-smoothed cursor position
     │
     ▼
 Canvas + UIOverlay   ← Draw strokes, blend onto frame, render HUD
     │
     ▼
 Display
```

**Key design decisions:**
- **Instant draw activation** — pinch is detected in 1 frame, zero lag
- **Sticky draw state** — requires 3 frames of clear unpinch to stop, preventing jitter mid-stroke
- **No direct DRAW → ERASE** — must pass through IDLE first, eliminating accidental erasing
- **Pinch midpoint cursor** — draws at the point between thumb and index tip for maximum stability

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Webcam not detected | Change `WEBCAM_INDEX = 0` to `1` in `air_draw.py` |
| Laggy performance | Lower `FRAME_WIDTH` / `FRAME_HEIGHT` in `air_draw.py` |
| Inaccurate tracking | Improve lighting, keep hand fully in frame |
| Drawing stops randomly | Pinch more firmly — thumb and index tips must be close |

---

## Tech Stack

- **[OpenCV](https://opencv.org/)** — Webcam capture, image processing, rendering
- **[MediaPipe](https://mediapipe.dev/)** — Real-time hand landmark detection (21 points)
- **[NumPy](https://numpy.org/)** — Array operations for canvas and coordinate math

---

## License

MIT

---

**Built by [@mAB-Barket](https://github.com/mAB-Barket)**
