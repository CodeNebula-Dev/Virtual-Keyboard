# Virtual Keyboard

**Type with your hands in mid-air — no physical keyboard required.**

Virtual Keyboard is a real-time, gesture-driven typing system that turns any standard webcam into an input device.  It uses computer-vision and machine-learning models to track your hands, overlays a full QWERTY keyboard on the camera feed, and translates a simple *pinch* gesture into actual key-press events that work in any application on your computer.

---

## Why This Project Exists

### The Problem

Millions of people around the world are **blind or visually impaired**.  The two most common digital-input methods available to them today are:

1. **Physical keyboards with screen readers** — effective, but requires memorising key positions and carrying a separate device.
2. **Speech recognition** — convenient in theory, but still far from perfect.  Background noise, accents, homophones, and specialised vocabulary (code, math, passwords) cause frequent mis-transcriptions.  Dictating punctuation and formatting commands is tedious.

Neither option is fully satisfying, especially in noisy environments, shared spaces, or when privacy matters (you may not want to say your password out loud).

### The Vision

Virtual Keyboard explores a **third path**: using **hand gestures captured by a camera** as the primary text-input method.

* **No special hardware** — any laptop webcam or USB camera works.
* **No sound needed** — works in complete silence, preserving privacy.
* **Tactile-free** — useful for people with motor impairments who find physical keys difficult, or for sterile environments (hospitals, labs).
* **Portable** — the camera is already built into your laptop; nothing extra to carry.

> **Note:**  This is a research prototype and proof of concept.  It is *not* yet a polished assistive-technology product.  The "Customisation Roadmap" section below lists the concrete steps needed to get there.

---

## How It Works — High-Level Architecture

```
Webcam Frame
    |
    v
[1] MediaPipe Hands  ──>  21 3-D Landmarks per hand
    |
    v
[2] Hand Tracking Module  ──>  Landmark list, bounding box, finger states
    |
    v
[3] Keyboard Engine  ──>  Overlay keyboard on frame
    |                      Detect pinch-over-key
    |                      Emit OS-level key-press via pynput
    v
[4] OpenCV Display  ──>  Annotated video feed shown to user
```

1. **Frame capture** — OpenCV grabs frames from the webcam at ~30 FPS.
2. **Hand detection and landmark regression** — MediaPipe Hands locates the hand and predicts 21 keypoints in 3-D.
3. **Gesture interpretation** — The keyboard engine checks whether the index-finger tip hovers over a virtual key *and* whether a pinch (index tip close to thumb tip) is happening simultaneously.
4. **Key emission** — When a press is confirmed (with debounce), `pynput` sends a real keystroke to the operating system, as if a physical key was pressed.

---

## Machine Learning Components

This project leverages several ML techniques under the hood, all provided by [Google MediaPipe](https://mediapipe.dev):

### 1. BlazePalm — Palm Detection Model

* **Architecture:** Single-shot detector (SSD) with a custom encoder.
* **Training data:** Approximately 18,000 annotated palm images covering diverse skin tones, lighting conditions, and hand orientations.
* **Output:** A bounding box around each detected palm, plus a coarse orientation estimate.
* **Why it matters:** Palm detection is the *first* stage.  It runs on every frame where the tracker has lost the hand, and it needs to be extremely fast (< 5 ms on CPU) to keep the pipeline real-time.

### 2. Hand Landmark Model — 21-Keypoint Regression

* **Architecture:** A lightweight convolutional neural network (CNN) that takes the cropped palm region as input and regresses **21 3-D keypoints** (x, y, z for each joint of every finger plus the wrist).
* **Coordinate system:** `x` and `y` are normalised to `[0, 1]` relative to the image dimensions.  `z` is a *relative depth* estimate (not metric) with the wrist as origin — fingers closer to the camera have more negative `z` values.
* **Real-time performance:** The model runs in < 10 ms per hand on a modern laptop CPU, enabling 30+ FPS even without a GPU.

### 3. Handedness Classification

* **What it does:** For each detected hand, MediaPipe outputs a classification label (`Left` or `Right`) along with a confidence score.
* **Use in this project:** The original codebase contained a two-hand variant (`Virtual_Keyboard/keyboard.py`) that mapped different characters to the left and right hands.  This handedness signal is what made that possible.

### 4. Tracking vs. Detection Trade-off

MediaPipe uses a **tracking-then-detection** strategy:

* On the *first* frame (or when tracking is lost), the heavier **palm detector** runs.
* On subsequent frames, a much cheaper **landmark tracker** follows the hand, skipping detection entirely.
* The `min_detection_confidence` and `min_tracking_confidence` thresholds (both configurable in this project) control when the system falls back from tracking to detection.

This design is what allows the system to be **real-time on CPU-only devices** — a critical requirement for accessibility tools that must run on low-end hardware.

---

## Project Structure

```
VirtualKeyboard/
    main.py               # Entry point — captures camera, orchestrates pipeline
    hand_tracking.py       # HandDetector class wrapping MediaPipe Hands
    keyboard_engine.py     # Virtual keyboard rendering and pinch-to-type logic
    list_cameras.py        # Utility to discover available camera indices
    requirements.txt       # Python dependencies
    .gitignore
    LICENSE
    README.md              # You are here
```

| File | Lines | Purpose |
|---|---|---|
| `main.py` | ~75 | CLI argument parsing, main loop, glue code |
| `hand_tracking.py` | ~175 | MediaPipe wrapper: detection, landmarks, finger states |
| `keyboard_engine.py` | ~180 | Key layout, overlay drawing, pinch detection, OS key emission |
| `list_cameras.py` | ~40 | Scan ports 0-9 and print which cameras are working |

---

## Getting Started

### Prerequisites

* **Python 3.9 or newer** (tested on 3.11 and 3.12)
* A working webcam (built-in or USB)
* macOS, Linux, or Windows

### Installation

```bash
# Clone the repo
git clone https://github.com/CodeNebula-Dev/VirtualKeyboard.git
cd VirtualKeyboard

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Default camera (index 0)
python main.py

# Specify a different camera
python main.py --camera 1

# Use an IP camera URL
python main.py --camera "http://192.168.1.10:8080/video"
```

### Discovering Cameras

If you are unsure which camera index to use:

```bash
python list_cameras.py
```

This scans ports 0-9 and prints which ones return frames.

### Controls

* **Pinch gesture** (bring index fingertip and thumb tip together while hovering over a key) — types that character.
* **Press `q`** on the OpenCV window — quits the application.

---

## Customisation Roadmap — Making It Better

The current codebase is a functional prototype.  Below are the concrete improvements and customisations needed to turn it into a robust, accessible tool.

### HIGH PRIORITY

#### 1. Audio Feedback for Visually Impaired Users

**Current state:** Visual-only feedback (key turns green on press).
**What to add:** Play a short audio cue (beep or spoken letter via `pyttsx3` / `gTTS`) every time a key is pressed.  This is *essential* for blind users who cannot see the screen.

**Where to modify:** `keyboard_engine.py` - `check_touches()` method, add an audio callback after each successful press.

#### 2. Backspace, Space, and Enter Keys

**Current state:** Only alphanumeric keys and basic punctuation are on the layout.
**What to add:** A bottom row with `SPACE`, `BACKSPACE`, `ENTER`, and `SHIFT` keys.  Map them to the corresponding `pynput.keyboard.Key` constants.

**Where to modify:** `keyboard_engine.py` - `DEFAULT_LAYOUT` list and `check_touches()` to handle special keys differently.

#### 3. Debounce and Accidental-Press Reduction

**Current state:** A simple time-based debounce (0.20 s).
**What to add:**
* Require the pinch to be held for at least 2-3 consecutive frames before firing (temporal smoothing).
* Add a "cooldown highlight" colour so the user knows a key was registered.

**Where to modify:** `keyboard_engine.py` - `check_touches()`.

#### 4. Depth-Based Press Detection

**Current state:** Press is detected purely by 2-D pinch distance.
**What to add:** Use MediaPipe's `z` coordinate (relative depth) to detect a "push forward" motion, simulating the feeling of pressing a physical key against a surface.  This would let users type by pushing their finger toward the camera / table instead of pinching.

**Where to modify:** `hand_tracking.py` - expose `z` values more prominently; `keyboard_engine.py` - add an alternative press-detection mode.

### MEDIUM PRIORITY

#### 5. Adaptive Key Sizing and Positioning

**Current state:** Fixed key size (60x60 px) and fixed grid origin.
**What to add:** Auto-scale the keyboard based on the detected hand size and camera resolution so it always fits comfortably within the user's reach.

**Where to modify:** `keyboard_engine.py` constructor and `draw()`.

#### 6. Multi-Hand Support with Distinct Roles

**Current state:** Only one hand is tracked for typing.
**What to add:** Track both hands simultaneously — e.g., left hand for modifiers (Shift, Ctrl, Alt) via fist gestures, right hand for typing.  The original `Virtual_Keyboard/keyboard.py` had an early version of this; the logic should be cleaned up and merged.

**Where to modify:** `main.py` loop (process both hands), `hand_tracking.py` (return per-hand data), `keyboard_engine.py` (accept hand label).

#### 7. Haptic / Vibration Feedback (Mobile)

**Current state:** Desktop-only, no haptic.
**What to add:** When deployed on a phone (e.g., via Kivy or a Flutter wrapper), trigger a short vibration on key press.

#### 8. Word Prediction and Auto-Complete

**Current state:** Character-by-character typing only.
**What to add:** Integrate a lightweight n-gram or transformer-based language model to suggest the next word, reducing the number of gestures needed.  Libraries like `symspellpy` or a small GPT-2 model could work.

### LOWER PRIORITY

#### 9. Custom Keyboard Layouts

**Current state:** Hardcoded QWERTY.
**What to add:** Load layouts from a JSON config file.  Support AZERTY, Dvorak, and custom accessibility layouts (e.g., most-frequent letters in the centre).

#### 10. On-Screen Text Display and Cursor

**Current state:** Typed text shown as a scrolling line at the bottom of the frame.
**What to add:** A proper text-area widget with a blinking cursor, line wrapping, and scroll support — rendered in OpenCV or via a small GUI (Tkinter / PyQt).

#### 11. Gesture Vocabulary Expansion

**Current state:** Only "pinch" is recognised.
**What to add:** Map additional gestures to actions:
  * Swipe left/right = move cursor
  * Open palm = space
  * Fist = backspace
  * Two-finger pinch = select word

#### 12. Performance Profiling and GPU Acceleration

**Current state:** CPU-only inference.
**What to add:** Profile the pipeline, identify bottlenecks, and optionally enable GPU inference via MediaPipe's GPU delegate or ONNX Runtime.

#### 13. Unit and Integration Tests

**Current state:** No automated tests.
**What to add:** Use `pytest` with pre-recorded landmark data (fixtures) to test gesture detection logic without a live camera.

---

## Code Quality Improvements Already Made

Compared to the original codebase, this repository includes the following clean-ups:

| Area | Before | After |
|---|---|---|
| **Naming** | `camelCase` mixed with `snake_case` | Consistent `snake_case` throughout |
| **Type hints** | None | All public methods annotated |
| **Docstrings** | Sparse inline comments | NumPy-style docstrings on every class and method |
| **Configuration** | Magic numbers scattered in code | Named constants and constructor parameters |
| **Debounce** | `time.sleep(0.15)` blocking the main loop | Non-blocking timestamp comparison |
| **Camera selection** | Hardcoded index `0` | CLI `--camera` flag supporting index or URL |
| **Security** | Password stored in source code | Removed entirely (was in the gesture-control script, not relevant here) |
| **Modularity** | Logic mixed across files | Clean separation: tracking / keyboard / main |

---

## Dependencies

| Library | Version | Role |
|---|---|---|
| [OpenCV](https://opencv.org/) | >= 4.8 | Camera capture, image processing, UI rendering |
| [MediaPipe](https://mediapipe.dev/) | >= 0.10 | Hand detection and 21-keypoint landmark ML model |
| [pynput](https://pynput.readthedocs.io/) | >= 1.7 | Emitting OS-level keyboard events |
| [NumPy](https://numpy.org/) | >= 1.24 | Array operations for the keyboard overlay |

---

## Contributing

Contributions are welcome!  If you'd like to help:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/audio-feedback`).
3. Commit your changes with clear messages.
4. Open a Pull Request describing what you changed and why.

Please keep the code style consistent (PEP 8, type hints, docstrings).

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

* [Google MediaPipe](https://mediapipe.dev/) for the hand-tracking ML models.
* The open-source computer-vision community for OpenCV.
* Accessibility researchers working on alternative input methods for people with disabilities.

---

*Built as a proof of concept to explore gesture-based typing as an accessible alternative to speech recognition and physical keyboards.*
