# ✋ Air Writing System

A real-time **Air Drawing** application built with Python, OpenCV, and MediaPipe.  
Point your finger at the webcam and draw in the air — no touchscreen needed!

---

## 🎮 Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ **Index finger only** | Draw with current color |
| ✌️ **Index + Middle** | Change color → **Red** |
| 🤟 **Index + Middle + Ring** | Change color → **Green** |
| 🖖 **Index + Middle + Ring + Pinky** | Change color → **Blue** |
| ✋ **Full open palm** | Eraser mode |
| **`C` key** | Clear the entire canvas |
| **`Q` key** | Quit the application |

---

## 🖥️ System Requirements

- Python **3.8** or higher
- A working **webcam**

---

## ⚙️ Installation

**Step 1 — Install Python** (if not already installed)  
Download from [python.org](https://www.python.org/downloads/)

**Step 2 — Install dependencies**

Open a terminal / command prompt in this project folder and run:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python mediapipe numpy
```

---

## ▶️ Running the App

```bash
python air_draw.py
```

A window titled **"Air Writing System"** will open showing your webcam feed.

> **Tip:** Make sure your hand is well-lit and clearly visible to the camera for best tracking accuracy.

---

## 📁 Project Structure

```
Air Writing/
├── air_draw.py        ← Main application (run this)
├── hand_tracker.py    ← MediaPipe hand landmark detection & finger logic
├── canvas.py          ← Virtual drawing canvas management
├── ui_overlay.py      ← HUD display (mode, color, gesture hints)
├── requirements.txt   ← Python dependencies
└── README.md          ← This file
```

---

## 🔬 How It Works

1. **MediaPipe Hands** detects 21 landmarks on your hand every frame.
2. `hand_tracker.py` compares each **fingertip Y-coordinate** to its **PIP knuckle Y-coordinate**.  
   - If the tip is *above* the knuckle (`tip_y < pip_y`) → finger is **UP**.
3. The combination of fingers that are up determines the **gesture**.
4. `canvas.py` maintains a NumPy array (same size as the webcam frame) where lines are drawn.
5. The canvas is **blended** onto the live webcam feed using bitmasking.
6. `ui_overlay.py` draws the semi-transparent HUD bars, finger indicators, and gesture hints.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| Webcam not opening | Change `WEBCAM_INDEX = 0` to `1` or `2` in `air_draw.py` |
| Laggy tracking | Reduce `FRAME_WIDTH`/`FRAME_HEIGHT` in `air_draw.py` |
| Finger detection inaccurate | Improve lighting; keep hand within frame |
| `mediapipe` install error | Try `pip install mediapipe==0.10.9` |

---

## 👨‍💻 Tech Stack

- [OpenCV](https://opencv.org/) — webcam capture & image processing  
- [MediaPipe](https://mediapipe.dev/) — real-time hand landmark detection  
- [NumPy](https://numpy.org/) — canvas array operations
