"""
Smart IR / Camera Gesture Controller — stable lightweight build (AI102L).

Labs 5–6: ``IRGestureDashboard`` encapsulates camera, CV, automation, and exports.
Lab 7: Pandas CSV export of gesture events.
Lab 8: Matplotlib session dashboard (Pandas export).

Core behaviour only: fist → screenshot; index → volume up; index+middle → volume down.
No handedness, brightness, diagnostics, or presence detection.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# --- timing / stability -------------------------------------------------------
GESTURE_WINDOW = 7
GESTURE_MIN_VOTES = 4
NO_HAND_HOLD_S = 0.25
HOLD_S = 0.55
COOLDOWN_S = 1.2
SCREENSHOT_COOLDOWN_S = 2.5
VOLUME_STEP_PCT = 5

# IR-safe stabilisation: absolute brightness thresholds mis-fire on dim IR streams
# (every frame looks “dark”), causing heavy flicker. We use EMA-relative dropouts.
DROPOUT_RATIO = 0.42
DROPOUT_MIN_DELTA = 6.0
DROPOUT_MAX_HOLD_FRAMES = 10
BRIGHTNESS_EMA_ALPHA = 0.12
EXPOSURE_JUMP_BLEND = 18.0
JUMP_BLEND_ALPHA = 0.55

CAP_W, CAP_H = 1280, 720

# MediaPipe hand topology (module-level: avoid rebuilding each session)
HAND_LINES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


def _mean_bgr(bgr: Any) -> float:
    """Scalar mean brightness without NumPy (OpenCV returns channel means)."""
    m = cv2.mean(bgr)
    return float(sum(m[:3]) / 3.0)


@dataclass
class GestureEvent:
    timestamp: str
    gesture: str
    action: str
    x: float
    y: float


class _ThreadedCapture:
    """Latest-frame reader thread (avoids OpenCV buffer lag / flicker)."""

    def __init__(self, cap: cv2.VideoCapture, flip: bool = True) -> None:
        self._cap = cap
        self._flip = flip
        self._run = threading.Event()
        self._run.set()
        self._lock = threading.Lock()
        self._latest: Any = None
        self._thr = threading.Thread(target=self._loop, daemon=True)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def start(self) -> None:
        self._thr.start()

    def _loop(self) -> None:
        while self._run.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.001)
                continue
            if self._flip:
                frame = cv2.flip(frame, 1)
            # Always store a copy — many V4L2 drivers reuse the same buffer; holding
            # a view causes torn / flashing frames when the reader overwrites it.
            with self._lock:
                self._latest = frame.copy()

    def read(self) -> Any:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def release(self) -> None:
        self._run.clear()
        self._thr.join(timeout=1.5)
        self._cap.release()


class _FrameStabilizer:
    """
    Suppress brief IR sensor dropouts without treating legitimately dim IR as “black”.

    - Dropout = sudden large dip vs exponential moving average of *good* means.
    - No full-time temporal blend (that amplifies auto-exposure pumping on IR).
    - Optional single-step blend only on rare brightness jumps.
    """

    def __init__(self) -> None:
        self._prev: Any = None
        self._prev_mean = 0.0
        self._bright_ema = 28.0
        self._drop_streak = 0

    def apply(self, bgr: Any) -> Any:
        m = _mean_bgr(bgr)
        ref = max(self._bright_ema, 8.0)

        # True black / lens covered — always a dropout if we have history.
        near_black = m < 3.5

        # Sudden dimming relative to what this stream normally looks like (IR-safe).
        relative_dim = (
            m < DROPOUT_RATIO * ref
            and (ref - m) >= DROPOUT_MIN_DELTA
        )

        is_dropout = (near_black or relative_dim) and self._prev is not None

        if is_dropout:
            self._drop_streak += 1
            if self._drop_streak <= DROPOUT_MAX_HOLD_FRAMES:
                return self._prev.copy()
            self._drop_streak = 0
            self._prev = None
            if m > 2.0:
                self._bright_ema = (1.0 - BRIGHTNESS_EMA_ALPHA) * self._bright_ema + BRIGHTNESS_EMA_ALPHA * m
            return bgr

        self._drop_streak = 0

        out = bgr
        if self._prev is not None:
            jump = abs(m - self._prev_mean)
            if jump >= EXPOSURE_JUMP_BLEND:
                out = cv2.addWeighted(
                    bgr, JUMP_BLEND_ALPHA, self._prev, 1.0 - JUMP_BLEND_ALPHA, 0
                )

        self._bright_ema = (1.0 - BRIGHTNESS_EMA_ALPHA) * self._bright_ema + BRIGHTNESS_EMA_ALPHA * m
        self._prev = out.copy()
        self._prev_mean = m
        return out


class _GestureVote:
    def __init__(self) -> None:
        self._q: deque[str] = deque(maxlen=GESTURE_WINDOW)
        self._stable = "OTHER"
        self._last_seen = 0.0

    def update(self, raw: str, has_hand: bool) -> str:
        now = time.time()
        if has_hand:
            self._last_seen = now
            self._q.append(raw)
            cand, votes = Counter(self._q).most_common(1)[0]
            if votes >= GESTURE_MIN_VOTES:
                self._stable = cand
        elif now - self._last_seen > NO_HAND_HOLD_S:
            self._q.clear()
            self._stable = "OTHER"
        return self._stable


class _ActionGate:
    def __init__(self) -> None:
        self._key: str | None = None
        self._t0 = 0.0
        self._last: dict[str, float] = {}

    def reset(self) -> None:
        self._key = None

    def fire(self, key: str | None) -> bool:
        if key is None:
            self.reset()
            return False
        now = time.time()
        if key != self._key:
            self._key = key
            self._t0 = now
            return False
        if now - self._t0 < HOLD_S:
            return False
        if now - self._last.get(key, 0.0) < COOLDOWN_S:
            return False
        self._last[key] = now
        self._t0 = now
        return True


def _v4l_indices() -> list[int]:
    nums: list[int] = []
    for p in sorted(Path("/dev").glob("video*")):
        s = p.name[5:]
        if s.isdigit():
            nums.append(int(s))
    return nums or [0, 1, 2, 3]


def _ir_score(frame: Any) -> float:
    b, g, r = cv2.split(frame)
    rg = cv2.mean(cv2.absdiff(r, g))[0]
    gb = cv2.mean(cv2.absdiff(g, b))[0]
    return float(rg + gb)


def _read_ok(cap: cv2.VideoCapture, n: int = 10) -> bool:
    for _ in range(n):
        ok, f = cap.read()
        if ok and f is not None and f.size > 0:
            return True
        time.sleep(0.05)
    return False


def _tune_capture_ir(cap: cv2.VideoCapture) -> None:
    """Best-effort V4L2 tweaks to reduce auto-exposure flicker on IR modules."""
    if sys.platform.startswith("win"):
        return
    for val in (0.25, 1.0, 0.75):
        try:
            if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val):
                break
        except Exception:
            continue
    try:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    except Exception:
        pass


def _open_camera() -> tuple[cv2.VideoCapture, str]:
    if sys.platform.startswith("win"):
        order = list(dict.fromkeys([1] + list(range(10))))
        backs = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        order = list(dict.fromkeys([1] + _v4l_indices()))
        backs = [cv2.CAP_V4L2]

    cand: list[tuple[float, int, int]] = []
    for be in backs:
        for idx in order:
            cap = cv2.VideoCapture(idx, be)
            try:
                if not cap.isOpened():
                    continue
                if not _read_ok(cap):
                    continue
                ok, fr = cap.read()
                if ok and fr is not None:
                    cand.append((_ir_score(fr), idx, be))
            finally:
                cap.release()
    cand.sort(key=lambda t: t[0])

    for sc, idx, be in cand:
        cap = cv2.VideoCapture(idx, be)
        if cap.isOpened() and _read_ok(cap, 12):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
            _tune_capture_ir(cap)
            return cap, f"index={idx} backend={be} ir_score≈{sc:.1f}"
        cap.release()

    cap = cv2.VideoCapture(0)
    if cap.isOpened() and _read_ok(cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        _tune_capture_ir(cap)
        return cap, "webcam index=0"
    cap.release()
    raise RuntimeError("Cannot open any camera.")


class IRGestureDashboard:
    """Single-class dashboard: capture → MediaPipe → gestures → CSV + plots."""

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.data_dir = self.root / "data"
        self.shots_dir = self.root / "screenshots"
        self.model_path = self.root / "hand_landmarker.task"
        self.csv_path = self.data_dir / "ir_session_data.csv"
        self.plot_path = self.data_dir / "ir_session_dashboard.png"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.shots_dir.mkdir(parents=True, exist_ok=True)

        self._events: list[GestureEvent] = []
        self._stab = _FrameStabilizer()
        self._vote = _GestureVote()
        self._gate = _ActionGate()
        self._shot_until = 0.0

        self._landmarker: mp_vision.HandLandmarker | None = None
        self._ts_ms = 0
        self._prev_t = time.perf_counter()

    def _ensure_model(self) -> None:
        if self.model_path.exists():
            return
        print("[..] Downloading hand_landmarker.task …")
        urllib.request.urlretrieve(MODEL_URL, str(self.model_path))
        print("[OK] Model ready.")

    def _init_mp(self) -> None:
        self._ensure_model()
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            num_hands=1,
            min_hand_detection_confidence=0.72,
            min_hand_presence_confidence=0.72,
            min_tracking_confidence=0.72,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    @staticmethod
    def _fingers(pts: list[tuple[int, int]]) -> tuple[int, list[int]]:
        if len(pts) < 21:
            return 0, []
        t = 1 if pts[4][0] < pts[3][0] else 0
        i = 1 if pts[8][1] < pts[6][1] else 0
        m = 1 if pts[12][1] < pts[10][1] else 0
        r = 1 if pts[16][1] < pts[14][1] else 0
        p = 1 if pts[20][1] < pts[18][1] else 0
        f = [t, i, m, r, p]
        return sum(f), f

    @staticmethod
    def _gesture(total: int, f: list[int]) -> str:
        if len(f) < 5:
            return "OTHER"
        t, i, m, r, p = f
        if total == 0:
            return "FIST"
        if t == 0 and i == 1 and m == 0 and r == 0 and p == 0:
            return "INDEX_UP"
        if t == 0 and i == 1 and m == 1 and r == 0 and p == 0:
            return "INDEX_MIDDLE_UP"
        return "OTHER"

    def _shot_cd_left(self) -> float:
        return max(0.0, self._shot_until - time.time())

    @staticmethod
    def _spawn(cmd: list[str]) -> None:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _screenshot(self, frame_bgr: Any) -> bool:
        """Non-blocking: ``subprocess.run`` would block the GUI; use ``Popen``."""
        if time.time() < self._shot_until:
            return False
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.shots_dir / f"screenshot_{ts}.png"
        try:
            if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and shutil.which(
                "grim"
            ):
                self._spawn(["grim", str(out)])
            elif shutil.which("gnome-screenshot"):
                self._spawn(["gnome-screenshot", "-f", str(out)])
            else:
                threading.Thread(
                    target=lambda: cv2.imwrite(str(out), frame_bgr.copy()),
                    daemon=True,
                ).start()
            self._shot_until = time.time() + SCREENSHOT_COOLDOWN_S
            print(f"[Screenshot] {out}")
            return True
        except Exception as exc:
            print(f"[Screenshot] error: {exc}")
            return False

    def _volume(self, direction: int) -> bool:
        label = "Up" if direction > 0 else "Down"
        step = VOLUME_STEP_PCT
        if shutil.which("xdotool"):
            key = (
                "XF86AudioRaiseVolume"
                if direction > 0
                else "XF86AudioLowerVolume"
            )
            for _ in range(2):
                try:
                    subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", key],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=4,
                    )
                    print(f"[Volume] {label} (xdotool)")
                    return True
                except Exception:
                    continue
        sign = "+" if direction > 0 else "-"
        for cmd in (
            ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{sign}{step}%"],
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{step}%"],
            ["amixer", "-q", "sset", "Master", f"{step}%{sign}"],
        ):
            if not shutil.which(cmd[0]):
                continue
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=6)
                print(f"[Volume] {label} ({cmd[0]})")
                return True
            except Exception as exc:
                print(f"[Volume] {cmd[0]}: {exc}")
        return False

    def _log(self, gesture: str, action: str, pts: list[tuple[int, int]]) -> None:
        if pts:
            n = len(pts)
            cx = sum(p[0] for p in pts) / n
            cy = sum(p[1] for p in pts) / n
        else:
            cx, cy = -1.0, -1.0
        self._events.append(
            GestureEvent(
                datetime.now().isoformat(timespec="seconds"),
                gesture,
                action,
                cx,
                cy,
            )
        )

    def _export(self) -> None:
        # Lazy import: keeps the realtime loop free of pandas/matplotlib startup cost.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        if not self._events:
            df = pd.DataFrame(
                columns=["timestamp", "gesture", "action", "x", "y"]
            )
        else:
            df = pd.DataFrame([e.__dict__ for e in self._events])
        df.to_csv(self.csv_path, index=False)
        print(f"[Data] CSV: {self.csv_path}")

        plt.figure(figsize=(11, 4.5))
        plt.subplot(1, 2, 1)
        if df.empty:
            plt.text(0.5, 0.5, "No events", ha="center", va="center")
            plt.axis("off")
        else:
            df["gesture"].value_counts().plot(kind="bar", color="#455a64")
            plt.title("Gesture counts")
            plt.grid(axis="y", alpha=0.3)

        plt.subplot(1, 2, 2)
        ok = df[(df["x"] >= 0) & (df["y"] >= 0)] if not df.empty else df
        if ok.empty:
            plt.text(0.5, 0.5, "No XY", ha="center", va="center")
            plt.axis("off")
        else:
            xs = ok["x"].tolist()
            ys = ok["y"].tolist()
            npt = len(xs)
            mx = sum(xs) / npt
            my = sum(ys) / npt
            dist = [
                ((xi - mx) ** 2 + (yi - my) ** 2) ** 0.5 for xi, yi in zip(xs, ys)
            ]
            plt.scatter(xs, ys, c=dist, cmap="viridis", s=40, alpha=0.85)
            plt.gca().invert_yaxis()
            plt.title("Hand centroid scatter")
            plt.grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=140)
        plt.close()
        print(f"[Data] Plot: {self.plot_path}")

    def run(self) -> None:
        print("IR Gesture Dashboard — Fist | Index | Index+Middle")
        self._init_mp()
        assert self._landmarker is not None

        cap, meta = _open_camera()
        print("[Camera]", meta)
        stream = _ThreadedCapture(cap)
        stream.start()

        try:
            while True:
                raw = stream.read()
                if raw is None:
                    time.sleep(0.002)
                    continue

                frame = self._stab.apply(raw)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if not rgb.flags["C_CONTIGUOUS"]:
                    rgb = rgb.copy(order="C")

                self._ts_ms += 1
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = self._landmarker.detect_for_video(mp_img, self._ts_ms)

                pts: list[tuple[int, int]] = []
                if res.hand_landmarks:
                    h, w = frame.shape[:2]
                    for lm in res.hand_landmarks[0]:
                        pts.append((int(lm.x * w), int(lm.y * h)))
                    for a, b in HAND_LINES:
                        if a < len(pts) and b < len(pts):
                            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2, cv2.LINE_AA)
                    for i, p in enumerate(pts):
                        cv2.circle(
                            frame,
                            p,
                            8 if i in (4, 8, 12, 16, 20) else 4,
                            (0, 180, 255) if i in (4, 8, 12, 16, 20) else (180, 180, 180),
                            -1,
                        )

                tot, fing = self._fingers(pts)
                raw_g = self._gesture(tot, fing)
                g = self._vote.update(raw_g, bool(pts))

                if g == "OTHER":
                    self._gate.reset()
                elif g == "FIST" and self._gate.fire("FIST"):
                    if self._screenshot(frame):
                        self._log("FIST", "screenshot", pts)
                elif g == "INDEX_UP" and self._gate.fire("IU"):
                    if self._volume(+1):
                        self._log("INDEX_UP", "volume_up", pts)
                elif g == "INDEX_MIDDLE_UP" and self._gate.fire("IM"):
                    if self._volume(-1):
                        self._log("INDEX_MIDDLE_UP", "volume_down", pts)

                # HUD
                fh, fw = frame.shape[:2]
                ovl = frame.copy()
                cv2.rectangle(ovl, (0, fh - 85), (fw, fh), (22, 22, 28), -1)
                cv2.addWeighted(ovl, 0.55, frame, 0.45, 0, frame)
                cv2.putText(
                    frame,
                    f"Gesture: {g}",
                    (12, fh - 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 220, 255),
                    2,
                )
                cd = self._shot_cd_left()
                if cd > 0:
                    cv2.putText(
                        frame,
                        f"COOLDOWN {cd:0.1f}s",
                        (fw - 280, fh - 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (60, 180, 255),
                        2,
                    )
                fps = 1.0 / max(time.perf_counter() - self._prev_t, 1e-6)
                self._prev_t = time.perf_counter()
                cv2.putText(
                    frame,
                    f"FPS {fps:.0f}",
                    (fw - 100, 28),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    (120, 255, 140),
                    1,
                )
                cv2.putText(
                    frame,
                    "Q quit",
                    (12, 28),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.1,
                    (180, 180, 180),
                    1,
                )

                cv2.imshow("IR Gesture Dashboard", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            try:
                stream.release()
            except Exception:
                pass
            try:
                if self._landmarker is not None:
                    self._landmarker.close()
            except Exception:
                pass
            cv2.destroyAllWindows()
            self._export()


def main() -> None:
    IRGestureDashboard().run()


if __name__ == "__main__":
    main()
