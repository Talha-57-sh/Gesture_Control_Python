# Smart IR / Camera Gesture Controller

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange)

Lightweight **AI102L** project: one hand, three gestures, session analytics.

## Features

| Gesture | Action |
|--------|--------|
| **Closed fist** (0 fingers up) | Screenshot → `screenshots/` (non-blocking helper process; **~2.5 s** on-screen cooldown) |
| **Index only** | Volume up (`xdotool` media key, else `wpctl` / `pactl` / `amixer`) |
| **Index + middle** | Volume down (same backends) |

- **Camera:** Picks the best IR-like V4L2 device when possible; falls back to index `0`.
- **Performance:** Background thread keeps only the **latest** frame (reduces buffer lag / flicker).
- **Labs 7 & 8:** On exit (`Q`), exports `data/ir_session_data.csv` and `data/ir_session_dashboard.png`.

## Run (Zorin / Ubuntu — use a venv)

```bash
cd "/path/to/python project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

Optional packages for screenshots / volume:

```bash
sudo apt install -y grim gnome-screenshot xdotool pipewire-audio-client pulseaudio-utils alsa-utils
```

Press **Q** in the OpenCV window to quit and generate analytics.

## AI102L mapping

- **Labs 5 & 6 (OOP):** Logic lives in `IRGestureDashboard` with small helper classes.
- **Lab 7 (Pandas):** Gesture log → CSV in `data/`.
- **Lab 8 (Matplotlib):** Centroid scatter + gesture bar chart → PNG in `data/`.

## Layout

```text
src/main.py          # full application
data/                # CSV + plot (generated)
screenshots/         # fist screenshots (generated)
hand_landmarker.task # auto-downloaded if missing
requirements.txt
README.md
```
