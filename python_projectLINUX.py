import os
import sys
import time
import subprocess
import urllib.request
import glob
from collections import Counter, deque
from datetime import datetime

# Windows fix: prevents MSMF camera-open issues on some laptops.
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
SCREENSHOT_DIR = os.path.join(os.getcwd(), "screenshots")
VOLUME_STEP_PERCENT = 5
HOLD_SECONDS = 0.6
COOLDOWN_SECONDS = 1.5
GESTURE_WINDOW_SIZE = 7
GESTURE_MIN_VOTES = 4
NO_HAND_HOLD_SECONDS = 0.25
# Try this index first for IR camera (commonly 1 on many laptops)
CAMERA_INDEX = 1

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

GESTURE_LABEL = {
    "FIST": "FIST -> Screenshot",
    "INDEX_UP": "INDEX -> Volume Up",
    "INDEX_MIDDLE_UP": "INDEX+MIDDLE -> Volume Down",
    "OPEN": "OPEN HAND (idle)",
    "OTHER": "No gesture detected",
}

GESTURE_COLOR = {
    "FIST": (0, 100, 255),
    "INDEX_UP": (0, 220, 0),
    "INDEX_MIDDLE_UP": (60, 60, 255),
    "OPEN": (200, 200, 200),
    "OTHER": (100, 100, 100),
}

try:
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
    VOLUME_AVAILABLE = True
    print("[OK] pycaw loaded.")
except Exception as e:
    VOLUME_AVAILABLE = False
    print(f"[!] pycaw unavailable: {e}")
    print("    Falling back to Linux audio commands.")


def ensure_model(path=MODEL_FILENAME):
    if os.path.exists(path):
        return path

    print(f"[..] Downloading model: {path}")

    def progress(block_num, block_size, total_size):
        pct = min(100, int(block_num * block_size / total_size * 100))
        print(f"\r     Progress: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, path, reporthook=progress)
    print("\n[OK] Model downloaded.")
    return path


def get_finger_state(landmarks):
    if len(landmarks) < 21:
        return 0, []

    thumb = 1 if landmarks[4][0] < landmarks[3][0] else 0
    index = 1 if landmarks[8][1] < landmarks[6][1] else 0
    middle = 1 if landmarks[12][1] < landmarks[10][1] else 0
    ring = 1 if landmarks[16][1] < landmarks[14][1] else 0
    pinky = 1 if landmarks[20][1] < landmarks[18][1] else 0
    fingers = [thumb, index, middle, ring, pinky]
    return sum(fingers), fingers


def detect_gesture(total, fingers):
    if len(fingers) < 5:
        return "OTHER"

    thumb, index, middle, ring, pinky = fingers

    if total == 0:
        return "FIST"
    if total == 5:
        return "OPEN"
    if thumb == 0 and index == 1 and middle == 0 and ring == 0 and pinky == 0:
        return "INDEX_UP"
    if thumb == 0 and index == 1 and middle == 1 and ring == 0 and pinky == 0:
        return "INDEX_MIDDLE_UP"
    return "OTHER"


class GestureStabilizer:
    """
    Smooth raw gesture predictions so IR flicker does not rapidly flip status.
    """

    def __init__(self, window_size=GESTURE_WINDOW_SIZE, min_votes=GESTURE_MIN_VOTES,
                 no_hand_hold=NO_HAND_HOLD_SECONDS):
        self.window_size = window_size
        self.min_votes = min_votes
        self.no_hand_hold = no_hand_hold
        self.history = deque(maxlen=window_size)
        self.stable_gesture = "OTHER"
        self.last_hand_seen = 0.0

    def update(self, raw_gesture, has_hand):
        now = time.time()
        if has_hand:
            self.last_hand_seen = now
            self.history.append(raw_gesture)
            counts = Counter(self.history)
            candidate, votes = counts.most_common(1)[0]
            if votes >= self.min_votes:
                self.stable_gesture = candidate
        else:
            # Keep previous stable gesture briefly when tracking drops for a few frames.
            if now - self.last_hand_seen > self.no_hand_hold:
                self.history.clear()
                self.stable_gesture = "OTHER"

        return self.stable_gesture


def take_screenshot():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.png")
    # Avoid importing pyautogui globally on Linux; it can fail without X auth.
    try:
        import pyautogui
        pyautogui.screenshot().save(path)
        print(f"[Screenshot] {path}")
        return
    except Exception:
        pass

    # Linux CLI fallbacks (works even when pyautogui is unavailable).
    for cmd in (["gnome-screenshot", "-f", path], ["scrot", path]):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[Screenshot] {path}")
            return
        except Exception:
            continue

    print("[Screenshot] Unavailable. Install one of: pyautogui, gnome-screenshot, scrot")


def change_volume(direction):
    if not VOLUME_AVAILABLE:
        delta = max(1, int(VOLUME_STEP_PERCENT))
        sign = "+" if direction > 0 else "-"

        # PipeWire (wpctl) first, then PulseAudio (pactl).
        fallback_cmds = [
            ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{sign}{delta}%"],
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{delta}%"],
        ]
        for cmd in fallback_cmds:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"[Volume] {'Up' if direction > 0 else 'Down'} by {delta}%")
                return
            except Exception:
                continue

        print("[Volume] Unavailable. Install/use wpctl or pactl.")
        return

    step = VOLUME_STEP_PERCENT / 100.0
    current = volume_ctrl.GetMasterVolumeLevelScalar()
    new_vol = float(np.clip(current + direction * step, 0.0, 1.0))
    volume_ctrl.SetMasterVolumeLevelScalar(new_vol, None)
    print(f"[Volume] {'Up' if direction > 0 else 'Down'} -> {int(new_vol * 100)}%")


def draw_landmarks(frame, lms):
    for a, b in HAND_CONNECTIONS:
        if a < len(lms) and b < len(lms):
            cv2.line(frame, lms[a], lms[b], (0, 200, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(lms):
        radius = 9 if i in (4, 8, 12, 16, 20) else 5
        color = (0, 180, 255) if i in (4, 8, 12, 16, 20) else (180, 180, 180)
        cv2.circle(frame, (x, y), radius, color, -1)


def draw_hud(frame, gesture, finger_count, action_label, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    label = GESTURE_LABEL.get(gesture, gesture)
    color = GESTURE_COLOR.get(gesture, (200, 200, 200))
    cv2.putText(frame, label, (12, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Fingers: {finger_count}", (12, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    if action_label:
        cv2.putText(frame, f"Action: {action_label}", (w - 260, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)
    return frame


def open_ir_camera():
    """
    Try to open an IR-like camera feed.
    We score frames by channel difference; lower usually means grayscale/IR.
    """
    if sys.platform.startswith("linux"):
        video_nodes = sorted(glob.glob("/dev/video*"))
        if not video_nodes:
            print("[ERROR] No camera device found at /dev/video*")
            print("        On Zorin/Linux: reconnect webcam, check BIOS camera setting,")
            print("        and ensure camera permissions are allowed.")
            return None, None, None, None

    indices = [CAMERA_INDEX] + list(range(0, 10))
    # remove duplicates while preserving order
    indices = list(dict.fromkeys(indices))
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    candidates = []

    def read_valid_frame(cap, attempts=12, delay=0.08):
        for _ in range(attempts):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return True, frame
            time.sleep(delay)
        return False, None

    for backend in backends:
        for idx in indices:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # Some camera drivers need warmup frames before first valid image.
            ok, frame = read_valid_frame(cap)
            if not ok:
                cap.release()
                continue

            b, g, r = cv2.split(frame)
            score = float(np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32))) +
                          np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32))))
            candidates.append((score, idx, backend))
            cap.release()

    if not candidates:
        return None, None, None, None

    # Lowest score is usually most IR-like (grayscale).
    candidates.sort(key=lambda x: x[0])

    # Re-open candidates in rank order. This avoids keeping test handles open.
    for score, idx, backend in candidates:
        for _ in range(3):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, frame = read_valid_frame(cap, attempts=15, delay=0.08)
                if ok:
                    return cap, idx, backend, score
            cap.release()
            time.sleep(0.2)

    return None, None, None, None


def main():
    print("Hand Gesture Controller")
    print("Fist -> Screenshot | Index -> Volume Up | Index+Middle -> Volume Down")

    model_path = ensure_model(MODEL_FILENAME)
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.75,
        min_tracking_confidence=0.75,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap, selected_idx, selected_backend, ir_score = open_ir_camera()
    if cap is None:
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"Camera index: {selected_idx}")
    print(f"Backend: {selected_backend}")
    if ir_score is not None:
        print(f"IR score (lower = more IR-like): {ir_score:.2f}")
    else:
        print("[Camera] IR feed not available, using normal camera fallback.")
    print(f"Screenshots: {SCREENSHOT_DIR}")

    current_gesture = None
    gesture_start = 0.0
    last_fired = {}
    stabilizer = GestureStabilizer()
    action_label = ""
    action_label_time = 0.0
    prev_time = time.time()
    timestamp_ms = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 1
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = []
        if result.hand_landmarks:
            h, w = frame.shape[:2]
            for lm in result.hand_landmarks[0]:
                landmarks.append((int(lm.x * w), int(lm.y * h)))
            draw_landmarks(frame, landmarks)

        finger_count, finger_state = get_finger_state(landmarks)
        raw_gesture = detect_gesture(finger_count, finger_state)
        gesture = stabilizer.update(raw_gesture, has_hand=bool(landmarks))

        now = time.time()
        if gesture != current_gesture:
            current_gesture = gesture
            gesture_start = now

        fired = None
        if now - gesture_start >= HOLD_SECONDS:
            if now - last_fired.get(gesture, 0) >= COOLDOWN_SECONDS:
                fired = gesture
                last_fired[gesture] = now
                gesture_start = now

        if fired == "FIST":
            take_screenshot()
            action_label = "Screenshot"
            action_label_time = now
        elif fired == "INDEX_UP":
            change_volume(+1)
            action_label = "Volume Up"
            action_label_time = now
        elif fired == "INDEX_MIDDLE_UP":
            change_volume(-1)
            action_label = "Volume Down"
            action_label_time = now

        if now - action_label_time > 2.0:
            action_label = ""

        curr = time.time()
        fps = 1.0 / max(curr - prev_time, 1e-6)
        prev_time = curr
        frame = draw_hud(frame, gesture, finger_count, action_label, fps)
        cv2.imshow("Gesture Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
