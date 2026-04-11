"""
ear_blink.py — Phase 2 (EAR) + Phase 8 (blink velocity for frustration)
Calculates Eye Aspect Ratio from FaceMesh landmarks.
Alerts if EAR < 0.25 for ≥ 2 seconds (drowsy/eyes closed).
Also tracks blink velocity for the frustration detector.
"""

import time
import threading
from scipy.spatial import distance as dist
import numpy as np


# ── Landmark indices for each eye (MediaPipe FaceMesh) ──────────────────────
#   Following the 6-point EAR formula:
#   p1-p6 going clockwise around the eye
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD      = 0.25   # below this = eye closing / closed
DROWSY_SECONDS     = 2.0    # seconds EAR must stay low before alert
BLINK_RATE_WINDOW  = 60     # seconds for blink-rate rolling window


def _eye_aspect_ratio(landmarks: list, eye_indices: list) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for one eye.
    landmarks : list of (x, y, z) pixel tuples for ALL 468 landmarks.
    eye_indices: 6 landmark indices [p1, p2, p3, p4, p5, p6].

    Formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    - Numerator: sum of two vertical distances
    - Denominator: horizontal distance (always large when eye is open)
    - Eye open  → EAR ≈ 0.30–0.40
    - Eye closed → EAR ≈ 0.05–0.15
    """
    pts = [np.array(landmarks[i][:2]) for i in eye_indices]  # (x, y) only
    # Vertical distances
    v1 = dist.euclidean(pts[1], pts[5])
    v2 = dist.euclidean(pts[2], pts[4])
    # Horizontal distance
    h  = dist.euclidean(pts[0], pts[3])
    ear = (v1 + v2) / (2.0 * h + 1e-6)   # 1e-6 prevents div-by-zero
    return ear


class BlinkDetector:
    """
    Reads landmarks from shared_state every ~33 ms (30 fps equivalent),
    computes EAR, detects blinks, tracks drowsiness, and writes results
    back to shared_state.

    shared_state keys written:
        ear               (float)  — current average EAR
        blink_count       (int)    — total blinks this session
        blink_rate_pm     (float)  — blinks per minute (last 60 s)
        blink_velocity    (float)  — speed of last blink (EAR drop rate)
        is_drowsy         (bool)   — True if EAR low ≥ 2 s
        drowsy_alert      (bool)   — one-shot flag; resets when eye opens
    """

    def __init__(self, shared_state: dict, fps: float = 30.0):
        self.shared_state   = shared_state
        self.fps            = fps
        self._stop_event    = threading.Event()

        # Internal tracking
        self._low_ear_start : float | None = None  # when EAR first went low
        self._blink_times   : list[float]  = []    # timestamps of blinks
        self._prev_ear      : float        = 0.3
        self._blink_open    : bool         = True   # eye was open last frame

    def stop(self):
        self._stop_event.set()

    def run(self):
        """Main loop — run in a dedicated thread."""
        interval = 1.0 / self.fps
        print("[BlinkDetector] Running.")

        while not self._stop_event.is_set():
            landmarks = self.shared_state.get("landmarks", [])

            if len(landmarks) < 468:
                # Face not detected — skip
                self.shared_state.update({
                    "ear": 0.0,
                    "is_drowsy": False,
                    "drowsy_alert": False,
                })
                time.sleep(interval)
                continue

            # ── Compute EAR ───────────────────────────────────────────────
            left_ear  = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
            right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
            ear       = (left_ear + right_ear) / 2.0

            # ── Blink detection (EAR drops below threshold then rises) ────
            blink_velocity = abs(ear - self._prev_ear)  # rate of EAR change

            if ear < EAR_THRESHOLD and self._blink_open:
                # Eyes just closed → start of blink
                self._blink_open    = False
                self._low_ear_start = time.time()
            elif ear >= EAR_THRESHOLD and not self._blink_open:
                # Eyes just opened → blink completed
                self._blink_open = True
                self._blink_times.append(time.time())
                self._low_ear_start = None   # reset drowsy timer

            # ── Drowsiness check ──────────────────────────────────────────
            is_drowsy     = False
            drowsy_alert  = False
            if self._low_ear_start is not None:
                elapsed = time.time() - self._low_ear_start
                if elapsed >= DROWSY_SECONDS:
                    is_drowsy    = True
                    drowsy_alert = True
                    print(f"[BlinkDetector] ⚠️  DROWSY ALERT! EAR={ear:.3f} "
                          f"for {elapsed:.1f}s")

            # ── Blink rate (per minute, rolling 60-second window) ─────────
            now = time.time()
            self._blink_times = [t for t in self._blink_times
                                  if now - t <= BLINK_RATE_WINDOW]
            blink_rate_pm = (len(self._blink_times) / BLINK_RATE_WINDOW) * 60

            # ── Write to shared state ─────────────────────────────────────
            self.shared_state.update({
                "ear"           : round(ear, 4),
                "blink_count"   : len(self._blink_times),
                "blink_rate_pm" : round(blink_rate_pm, 2),
                "blink_velocity": round(blink_velocity, 4),
                "is_drowsy"     : is_drowsy,
                "drowsy_alert"  : drowsy_alert,
            })

            self._prev_ear = ear
            time.sleep(interval)

        print("[BlinkDetector] Stopped.")


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from face_tracker import FaceTracker

    state = {}
    tracker  = FaceTracker(shared_state=state)
    detector = BlinkDetector(shared_state=state)

    t = threading.Thread(target=detector.run, daemon=True)
    t.start()

    tracker.run()  # blocking
    detector.stop()
