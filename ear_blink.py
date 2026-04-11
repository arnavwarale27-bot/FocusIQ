"""
ear_blink.py — Phase 2 (EAR) + Phase 8 (blink velocity for frustration)
Blink health targets:
  Healthy   : 12-18 blinks/min
  Low alert : < 8 blinks/min  -> DROWSY (staring / fatigued)
  Closed alert: EAR < threshold for >= 2s -> DROWSY (eyes shut)

Bug fixes (v2):
  1. Landmark guard uses MAX_IDX_NEEDED (387) not hardcoded 468,
     because refine_landmarks=True produces 478 points.
  2. Drowsy timer and blink timer are separate variables - a quick
     blink resets the blink counter but NOT the closed-eye drowsy timer.
  3. Blink rate uses actual elapsed time, not a fixed 60s denominator.
  4. Added low-blink-rate drowsy signal (< 8 blinks/min after 30s uptime).
  5. Auto-calibration of EAR threshold per user (first 3 seconds).
"""

import time
import threading
from scipy.spatial import distance as dist
import numpy as np


# Landmark indices (MediaPipe FaceMesh, 6 points per eye)
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# The highest landmark index needed across both eyes
MAX_IDX_NEEDED = max(max(LEFT_EYE_IDX), max(RIGHT_EYE_IDX))  # = 387

# Thresholds
EAR_THRESHOLD        = 0.22   # default; overridden by auto-calibration
BLINK_MAX_DURATION   = 0.40   # blinks longer than this = eyes-closed event
DROWSY_CLOSED_SECS   = 2.0    # eyes closed this long -> DROWSY
BLINK_RATE_WINDOW    = 60.0   # rolling window (seconds)
LOW_BLINK_THRESHOLD  = 8.0    # blinks/min below this -> DROWSY
LOW_BLINK_MIN_UPTIME = 30.0   # wait this long before low-blink alert fires


def _eye_aspect_ratio(landmarks: list, eye_indices: list) -> float:
    """
    EAR formula (Soukupova & Cech, 2016):
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Point layout (looking at the face):
        p1 -------- p4     <- horizontal (eye width)
          p2    p3          <- upper eyelid
          p6    p5          <- lower eyelid

    Open eye : EAR ~ 0.28-0.38
    Closed   : EAR ~ 0.04-0.15
    """
    try:
        pts = [np.array(landmarks[i][:2], dtype=np.float64) for i in eye_indices]
    except (IndexError, TypeError):
        return 0.0

    v1 = dist.euclidean(pts[1], pts[5])  # ||p2-p6||
    v2 = dist.euclidean(pts[2], pts[4])  # ||p3-p5||
    h  = dist.euclidean(pts[0], pts[3])  # ||p1-p4||

    if h < 1e-3:        # eye is off-frame or degenerate
        return 0.0

    return float((v1 + v2) / (2.0 * h))


class BlinkDetector:
    """
    Reads landmarks from shared_state, computes EAR, detects blinks,
    and writes results back. Runs in its own daemon thread.

    shared_state keys WRITTEN:
        ear               float   average EAR both eyes
        left_ear          float   left eye EAR
        right_ear         float   right eye EAR
        blink_count       int     blinks in current rolling window
        blink_rate_pm     float   blinks/min (rolling 60s)
        blink_velocity    float   |delta EAR| per frame
        is_drowsy         bool    True if either drowsy signal fires
        drowsy_reason     str     'eyes_closed' | 'low_blink_rate' | ''
        drowsy_alert      bool    alias for is_drowsy
        session_uptime    float   seconds since start
    """

    def __init__(self, shared_state: dict, fps: float = 30.0):
        self.shared_state  = shared_state
        self.fps           = fps
        self._stop_event   = threading.Event()
        self._start_time   = time.time()

        # Blink state machine
        self._eye_open          = True
        self._eye_closed_since  = None   # when eye closed (for drowsy timer)
        self._blink_times       = []     # timestamps of completed blinks

        # EAR history
        self._prev_ear          = 0.30

        # Per-user calibration (first 90 frames ~ 3s)
        self._cal_ears          = []
        self._calibrated        = False
        self._ear_threshold     = EAR_THRESHOLD

    def stop(self):
        self._stop_event.set()

    # ------------------------------------------------------------------
    def _calibrate(self, ear: float):
        if self._calibrated:
            return
        if ear > 0.15:
            self._cal_ears.append(ear)
        if len(self._cal_ears) >= 90:
            arr = np.array(self._cal_ears)
            t = float(np.mean(arr) - 1.5 * np.std(arr))
            self._ear_threshold = max(0.12, min(0.25, t))
            self._calibrated = True
            print(f"[BlinkDetector] Calibrated: threshold={self._ear_threshold:.3f} "
                  f"(mean={np.mean(arr):.3f})")

    # ------------------------------------------------------------------
    def _update_blink_state(self, ear: float, now: float) -> float:
        """
        State machine: track open/close transitions.
        Returns blink_velocity (|delta EAR|).
        Short close (<= 0.4s) -> blink.
        Long close  (>  0.4s) -> eyes-closed event (handled by drowsy check).
        """
        velocity = abs(ear - self._prev_ear)
        threshold = self._ear_threshold

        if ear < threshold:
            if self._eye_open:
                # --- Transition: open -> closed ---
                self._eye_open         = False
                self._eye_closed_since = now
        else:
            if not self._eye_open:
                # --- Transition: closed -> open ---
                closed_dur = now - (self._eye_closed_since or now)
                if closed_dur <= BLINK_MAX_DURATION:
                    # Short close = blink
                    self._blink_times.append(now)
                    print(f"[BlinkDetector] Blink! "
                          f"dur={closed_dur*1000:.0f}ms EAR={ear:.3f}")
                # Reset eye state but keep _eye_closed_since for drowsy check
                self._eye_open = True
                # IMPORTANT: do NOT reset _eye_closed_since here.
                # drowsy check reads it independently.

        return velocity

    # ------------------------------------------------------------------
    def _check_drowsy(self, ear: float, blink_rate_pm: float, now: float):
        """Returns (is_drowsy: bool, reason: str)."""
        # Signal A: eyes continuously closed
        if not self._eye_open and self._eye_closed_since is not None:
            closed_for = now - self._eye_closed_since
            if closed_for >= DROWSY_CLOSED_SECS:
                print(f"[BlinkDetector] DROWSY (eyes closed {closed_for:.1f}s) "
                      f"EAR={ear:.3f}")
                return True, "eyes_closed"

        # Signal B: blink rate too low (only after warm-up)
        uptime = now - self._start_time
        if uptime >= LOW_BLINK_MIN_UPTIME and blink_rate_pm < LOW_BLINK_THRESHOLD:
            print(f"[BlinkDetector] DROWSY (low blink rate {blink_rate_pm:.1f}/min)")
            return True, "low_blink_rate"

        return False, ""

    # ------------------------------------------------------------------
    def run(self):
        interval = 1.0 / self.fps
        print("[BlinkDetector] Running.")

        while not self._stop_event.is_set():
            landmarks = self.shared_state.get("landmarks", [])

            # Need landmarks up to index 387 at minimum
            if len(landmarks) <= MAX_IDX_NEEDED:
                self.shared_state.update({
                    "ear": 0.0, "left_ear": 0.0, "right_ear": 0.0,
                    "is_drowsy": False, "drowsy_reason": "", "drowsy_alert": False,
                })
                time.sleep(interval)
                continue

            now = time.time()

            # Compute EAR for both eyes
            left_ear  = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
            right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
            ear       = (left_ear + right_ear) / 2.0

            # Skip degenerate frames
            if left_ear == 0.0 and right_ear == 0.0:
                time.sleep(interval)
                continue

            # Auto-calibrate threshold (first 3s)
            self._calibrate(ear)

            # Blink state machine
            blink_velocity = self._update_blink_state(ear, now)

            # Rolling blink rate using actual elapsed time
            self._blink_times = [t for t in self._blink_times
                                  if now - t <= BLINK_RATE_WINDOW]
            elapsed = min(now - self._start_time, BLINK_RATE_WINDOW)
            blink_rate_pm = (len(self._blink_times) / elapsed * 60.0
                             if elapsed > 0 else 0.0)

            # Drowsiness check
            is_drowsy, drowsy_reason = self._check_drowsy(ear, blink_rate_pm, now)

            self.shared_state.update({
                "ear"           : round(ear,           4),
                "left_ear"      : round(left_ear,      4),
                "right_ear"     : round(right_ear,     4),
                "blink_count"   : len(self._blink_times),
                "blink_rate_pm" : round(blink_rate_pm, 2),
                "blink_velocity": round(blink_velocity, 4),
                "is_drowsy"     : is_drowsy,
                "drowsy_reason" : drowsy_reason,
                "drowsy_alert"  : is_drowsy,
                "session_uptime": round(now - self._start_time, 1),
            })

            self._prev_ear = ear
            time.sleep(interval)

        print("[BlinkDetector] Stopped.")


# Standalone test
if __name__ == "__main__":
    import os, sys
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_tracker import FaceTracker

    state    = {}
    detector = BlinkDetector(shared_state=state)
    t        = threading.Thread(target=detector.run, daemon=True)
    t.start()

    def _stats():
        while True:
            time.sleep(1)
            print(f"  EAR={state.get('ear',0):.3f} "
                  f"L={state.get('left_ear',0):.3f} "
                  f"R={state.get('right_ear',0):.3f} | "
                  f"Blinks={state.get('blink_count',0)} "
                  f"Rate={state.get('blink_rate_pm',0):.1f}/min | "
                  f"Drowsy={state.get('is_drowsy',False)} "
                  f"({state.get('drowsy_reason','')})")

    threading.Thread(target=_stats, daemon=True).start()
    FaceTracker(shared_state=state).run()
    detector.stop()
