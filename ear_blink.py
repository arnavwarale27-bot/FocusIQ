"""
ear_blink.py — Blink detection via Eye Aspect Ratio (EAR)

Uses MediaPipe Face Mesh landmark indices to compute EAR for both eyes.
- Normal blinks (EAR < 0.25 for < 400ms) are counted but NOT penalized
- Drowsy = True only when eyes continuously closed > 2 seconds
- Rolling 60s blink rate tracked via deque of timestamps

shared_state keys READ:
    landmarks       list of (x, y, z) tuples — pixel coords from FaceTracker

shared_state keys WRITTEN:
    ear             float   average EAR across both eyes
    blink_count     int     total blinks detected
    blink_rate_pm   float   blinks per minute (rolling 60s window)
    drowsy          bool    True if eyes closed > 2 seconds continuously
"""

import time
import threading
import collections
import numpy as np
from scipy.spatial import distance as dist


# ── MediaPipe Face Mesh landmark indices ─────────────────────────────────────
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

MAX_IDX_NEEDED = max(max(LEFT_EYE_IDX), max(RIGHT_EYE_IDX))  # 387

# ── Thresholds ───────────────────────────────────────────────────────────────
EAR_THRESHOLD       = 0.25   # below this = eyes considered closed
BLINK_MAX_DURATION  = 0.40   # 400ms — anything shorter is a normal blink
DROWSY_SECONDS      = 2.0    # eyes closed this long = drowsy


def _eye_aspect_ratio(landmarks: list, eye_indices: list) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    try:
        pts = [np.array([landmarks[i][0], landmarks[i][1]], dtype=np.float64)
               for i in eye_indices]
    except (IndexError, TypeError):
        return 0.0

    v1 = dist.euclidean(pts[1], pts[5])  # ||p2 - p6||
    v2 = dist.euclidean(pts[2], pts[4])  # ||p3 - p5||
    h  = dist.euclidean(pts[0], pts[3])  # ||p1 - p4||

    if h < 1e-6:
        return 0.0

    return float((v1 + v2) / (2.0 * h))


class BlinkDetector:
    """
    Reads landmarks from shared_state, computes EAR, detects blinks
    and drowsiness. Runs in its own daemon thread.
    """

    def __init__(self, shared_state: dict, fps: float = 30.0):
        self.shared_state = shared_state
        self.fps = fps
        self._stop_event = threading.Event()

        # Eye state tracking
        self._eye_open = True
        self._eye_closed_since = None   # timestamp when eyes first closed

        # Blink counting
        self._blink_count = 0
        self._blink_times: collections.deque = collections.deque()  # rolling 60s

    def stop(self):
        self._stop_event.set()

    def run(self):
        interval = 1.0 / self.fps
        print("[BlinkDetector] Running.")

        while not self._stop_event.is_set():
            landmarks = self.shared_state.get("landmarks", [])

            if len(landmarks) <= MAX_IDX_NEEDED:
                self.shared_state["ear"] = 0.0
                self.shared_state["drowsy"] = False
                time.sleep(interval)
                continue

            now = time.time()

            # ── Compute EAR ──────────────────────────────────────────────
            left_ear  = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
            right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
            ear = (left_ear + right_ear) / 2.0

            # ── State machine: open ↔ closed transitions ────────────────
            if ear < EAR_THRESHOLD:
                # Eyes are below threshold
                if self._eye_open:
                    # Transition: open → closed
                    self._eye_open = False
                    self._eye_closed_since = now
            else:
                # Eyes are above threshold
                if not self._eye_open:
                    # Transition: closed → open
                    closed_duration = now - (self._eye_closed_since or now)

                    if closed_duration <= BLINK_MAX_DURATION:
                        # Normal blink (< 400ms) — count it, no penalty
                        self._blink_count += 1
                        self._blink_times.append(now)
                        print(f"[BlinkDetector] Blink #{self._blink_count}  "
                              f"dur={closed_duration*1000:.0f}ms  EAR={ear:.3f}")

                    # Reset eye state
                    self._eye_open = True
                    self._eye_closed_since = None

            # ── Rolling blink rate (60s window) ──────────────────────────
            while self._blink_times and (now - self._blink_times[0]) > 60.0:
                self._blink_times.popleft()
            blink_rate_pm = len(self._blink_times)

            # ── Drowsiness: eyes continuously closed > 2 seconds ─────────
            drowsy = False
            if not self._eye_open and self._eye_closed_since is not None:
                closed_for = now - self._eye_closed_since
                if closed_for > DROWSY_SECONDS:
                    drowsy = True
                    print(f"[BlinkDetector] DROWSY — eyes closed "
                          f"{closed_for:.1f}s  EAR={ear:.3f}")

            # ── Write to shared state ────────────────────────────────────
            self.shared_state["ear"] = round(ear, 4)
            self.shared_state["blink_count"] = self._blink_count
            self.shared_state["blink_rate_pm"] = blink_rate_pm
            self.shared_state["drowsy"] = drowsy

            time.sleep(interval)

        print("[BlinkDetector] Stopped.")


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_tracker import FaceTracker

    state = {}
    detector = BlinkDetector(shared_state=state)
    t = threading.Thread(target=detector.run, daemon=True)
    t.start()

    def _stats():
        while True:
            time.sleep(1)
            print(f"  EAR={state.get('ear', 0):.3f} | "
                  f"Blinks={state.get('blink_count', 0)} "
                  f"({state.get('blink_rate_pm', 0)}/min) | "
                  f"Drowsy={state.get('drowsy', False)}")

    threading.Thread(target=_stats, daemon=True).start()
    FaceTracker(shared_state=state).run()
    detector.stop()
