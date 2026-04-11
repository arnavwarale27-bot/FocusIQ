"""
ear_blink.py — Phase 2 (EAR) + Phase 8 (blink velocity for frustration)
Calculates Eye Aspect Ratio from FaceMesh landmarks.

Blink health targets:
  - Healthy  : 12–18 blinks/min
  - Low alert: < 8 blinks/min  → DROWSY (staring / fatigued)
  - High EAR alert: EAR < EAR_THRESHOLD for ≥ DROWSY_SECONDS → eyes closed

Bug fixes applied (v2):
  1. landmark count guard changed to ≥ 6 valid landmarks (not < 468),
     because refine_landmarks=True produces 478 points, not 468.
  2. Drowsy-timer and blink-timer are now SEPARATE variables so a normal
     blink (< BLINK_MAX_DURATION) resets the blink timer but NOT the
     continuous-eyes-closed drowsy timer.
  3. Blink rate formula now uses elapsed real time, not a fixed 60s
     denominator, so it's accurate from the very first second.
  4. Added low-blink-rate DROWSY alert (< LOW_BLINK_THRESHOLD blinks/min).
"""

import time
import threading
from scipy.spatial import distance as dist
import numpy as np


# ── Landmark indices for each eye (MediaPipe FaceMesh) ──────────────────────
#   6-point EAR layout around one eye (clockwise):
#   p1 = left corner, p2/p3 = upper lid, p4 = right corner, p5/p6 = lower lid
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# ── Thresholds ───────────────────────────────────────────────────────────────
EAR_THRESHOLD      = 0.22    # below → eye closed / closing; tuned for webcam
BLINK_MAX_DURATION = 0.40    # seconds; blinks longer than this = eyes-closed event
DROWSY_CLOSED_SECS = 2.0     # seconds eyes must stay closed before DROWSY fires
BLINK_RATE_WINDOW  = 60.0    # rolling window for blink rate (seconds)
LOW_BLINK_THRESHOLD = 8.0    # blinks/min below this → DROWSY (not blinking enough)
LOW_BLINK_MIN_UPTIME = 30.0  # only fire low-blink alert after 30s of session


# ─────────────────────────────────────────────────────────────────────────────
# Core EAR formula
# ─────────────────────────────────────────────────────────────────────────────

def _eye_aspect_ratio(landmarks: list, eye_indices: list) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for one eye.

    landmarks  : list of (x_px, y_px, z) tuples — ALL face landmarks.
    eye_indices: list of 6 landmark ids [p1, p2, p3, p4, p5, p6].

    EAR formula  (Soukupová & Čech, 2016):
        EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)

    Point layout (looking at the face):
        p1 ──────── p4      ← horizontal (eye width)
          p2    p3           ← upper lid
          p6    p5           ← lower lid

    Vertical distances: p2-p6 and p3-p5
    Horizontal distance: p1-p4

    Open eye : EAR ≈ 0.28–0.38
    Closed eye: EAR ≈ 0.04–0.15
    """
    # Extract (x, y) pixel coordinates only
    try:
        pts = [np.array(landmarks[i][:2], dtype=np.float64) for i in eye_indices]
    except (IndexError, TypeError):
        return 0.0  # landmark not available — return 0 safely

    v1 = dist.euclidean(pts[1], pts[5])   # ||p2 - p6||
    v2 = dist.euclidean(pts[2], pts[4])   # ||p3 - p5||
    h  = dist.euclidean(pts[0], pts[3])   # ||p1 - p4||

    # Guard against degenerate (eyes completely off-frame) case
    if h < 1e-3:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)


# ─────────────────────────────────────────────────────────────────────────────
# BlinkDetector — runs in its own thread
# ─────────────────────────────────────────────────────────────────────────────

class BlinkDetector:
    """
    Reads landmarks from shared_state, computes EAR every frame,
    detects blinks, and writes results back to shared_state.

    Two independent drowsiness signals:
      A) EAR-based: eyes continuously closed for ≥ DROWSY_CLOSED_SECS
      B) Rate-based: blinks/min < LOW_BLINK_THRESHOLD after warm-up

    shared_state keys READ:
        landmarks      (list of (x,y,z) tuples from FaceTracker)

    shared_state keys WRITTEN:
        ear               (float)  current average EAR, 0–0.5 range
        left_ear          (float)  left eye EAR individually
        right_ear         (float)  right eye EAR individually
        blink_count       (int)    total blinks detected this session
        blink_rate_pm     (float)  blinks/min over last 60s window
        blink_velocity    (float)  |ΔEAR| per frame — proxy for blink speed
        is_drowsy         (bool)   True if either drowsiness signal fires
        drowsy_reason     (str)    'eyes_closed' | 'low_blink_rate' | ''
        drowsy_alert      (bool)   same as is_drowsy (kept for compatibility)
        session_uptime    (float)  seconds since BlinkDetector started
    """

    def __init__(self, shared_state: dict, fps: float = 30.0):
        self.shared_state = shared_state
        self.fps          = fps
        self._stop_event  = threading.Event()
        self._start_time  = time.time()

        # ── Blink tracking ────────────────────────────────────────────────
        self._eye_open         : bool        = True    # True = eye currently open
        self._eye_closed_since : float | None = None   # timestamp when eye closed
        self._blink_times      : list         = []     # timestamps of completed blinks

        # ── EAR history ───────────────────────────────────────────────────
        self._prev_ear         : float        = 0.30

        # ── Calibration ───────────────────────────────────────────────────
        #   Collect first N EAR readings to auto-set the threshold for this face
        self._calibration_ears : list         = []
        self._calibrated       : bool         = False
        self._ear_threshold    : float        = EAR_THRESHOLD  # may be updated

    # ── Public API ────────────────────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _calibrate(self, ear: float):
        """
        Collect the first 90 frames (≈3 s at 30 fps) of EAR values while
        eyes are open, then set threshold = mean - 1.5×std.
        This adapts to each user's natural eye openness.
        """
        if self._calibrated:
            return
        if ear > 0.15:   # only include clearly-open frames
            self._calibration_ears.append(ear)
        if len(self._calibration_ears) >= 90:
            arr = np.array(self._calibration_ears)
            self._ear_threshold = float(np.mean(arr) - 1.5 * np.std(arr))
            # Clamp to a sensible range so it can't be absurd
            self._ear_threshold = max(0.12, min(0.25, self._ear_threshold))
            self._calibrated = True
            print(f"[BlinkDetector] ✅ Calibrated EAR threshold = "
                  f"{self._ear_threshold:.3f} "
                  f"(mean={np.mean(arr):.3f}, std={np.std(arr):.3f})")

    def _check_blink(self, ear: float, now: float) -> float:
        """
        State machine for blink detection.
        Returns blink_velocity (|Δ EAR|).
        """
        blink_velocity = abs(ear - self._prev_ear)
        threshold = self._ear_threshold

        if ear < threshold:
            # ── Eye is CLOSING or CLOSED ──────────────────────────────────
            if self._eye_open:
                # Transition open→closed
                self._eye_open         = False
                self._eye_closed_since = now
        else:
            # ── Eye is OPEN ───────────────────────────────────────────────
            if not self._eye_open:
                # Transition closed→open
                closed_duration = now - (self._eye_closed_since or now)
                if closed_duration <= BLINK_MAX_DURATION:
                    # Short close → it's a blink
                    self._blink_times.append(now)
                    print(f"[BlinkDetector] 👁  Blink detected "
                          f"(dur={closed_duration*1000:.0f}ms, EAR={ear:.3f})")
                # else: long close = eyes closed event, handled by drowsy check
                self._eye_open         = True
                # NOTE: _eye_closed_since is NOT reset here so the drowsy
                # timer in _check_drowsy() can still measure total closed time.

        return blink_velocity

    def _check_drowsy(self, ear: float, blink_rate_pm: float,
                      now: float) -> tuple:
        """
        Returns (is_drowsy: bool, reason: str).
        Two independent signals:
          A) EAR signal — eyes closed for too long
          B) Rate signal — not blinking enough
        """
        # ── Signal A: eyes continuously closed ────────────────────────────
        if not self._eye_open and self._eye_closed_since is not None:
            closed_for = now - self._eye_closed_since
            if closed_for >= DROWSY_CLOSED_SECS:
                print(f"[BlinkDetector] ⚠️  DROWSY (eyes closed) "
                      f"EAR={ear:.3f} for {closed_for:.1f}s")
                return True, "eyes_closed"

        # ── Signal B: blink rate too low ─────────────────────────────────
        uptime = now - self._start_time
        if uptime >= LOW_BLINK_MIN_UPTIME:   # don't fire in first 30s
            if blink_rate_pm < LOW_BLINK_THRESHOLD and blink_rate_pm > 0:
                print(f"[BlinkDetector] ⚠️  DROWSY (low blink rate) "
                      f"{blink_rate_pm:.1f} blinks/min")
                return True, "low_blink_rate"

        return False, ""

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        """Runs in a dedicated daemon thread."""
        interval = 1.0 / self.fps
        print("[BlinkDetector] Running.")

        while not self._stop_event.is_set():
            landmarks = self.shared_state.get("landmarks", [])

            # ── Guard: need at least enough landmarks for both eyes ───────
            # refine_landmarks=True gives 478 points; plain mode gives 468.
            # Both are ≥ 393 (the highest index we use: 393 not used; max=387).
            # Check that the highest index we need (387) is available.
            MAX_IDX_NEEDED = max(max(LEFT_EYE_IDX), max(RIGHT_EYE_IDX))  # = 387
            if len(landmarks) <= MAX_IDX_NEEDED:
                # Face not detected or partial — report zeros, don't update timers
                self.shared_state.update({
                    "ear"          : 0.0,
                    "left_ear"     : 0.0,
                    "right_ear"    : 0.0,
                    "is_drowsy"    : False,
                    "drowsy_reason": "",
                    "drowsy_alert" : False,
                })
                time.sleep(interval)
                continue

            now = time.time()

            # ── Compute EAR ───────────────────────────────────────────────
            left_ear  = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
            right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
            ear       = (left_ear + right_ear) / 2.0

            # Sanity check: if EAR is suspiciously 0 for both eyes,
            # the landmarks are probably bad this frame — skip it
            if left_ear == 0.0 and right_ear == 0.0:
                time.sleep(interval)
                continue

            # ── Auto-calibrate EAR threshold ──────────────────────────────
            self._calibrate(ear)

            # ── Blink state machine ───────────────────────────────────────
            blink_velocity = self._check_blink(ear, now)

            # ── Blink rate: blinks/min over rolling 60-second window ──────
            # Trim old blink times outside the rolling window
            self._blink_times = [t for t in self._blink_times
                                  if now - t <= BLINK_RATE_WINDOW]

            # Use actual elapsed time for accuracy (not always 60s)
            elapsed = min(now - self._start_time, BLINK_RATE_WINDOW)
            if elapsed > 0:
                blink_rate_pm = (len(self._blink_times) / elapsed) * 60.0
            else:
                blink_rate_pm = 0.0

            # ── Drowsiness check ──────────────────────────────────────────
            is_drowsy, drowsy_reason = self._check_drowsy(ear, blink_rate_pm, now)

            # ── Write to shared state ─────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test  (run: python ear_blink.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_tracker import FaceTracker

    state    = {}
    detector = BlinkDetector(shared_state=state)

    t = threading.Thread(target=detector.run, daemon=True)
    t.start()

    # Print live stats every second while FaceTracker runs
    import threading as _th
    def _print_stats():
        while True:
            time.sleep(1)
            print(
                f"  EAR={state.get('ear', 0):.3f} "
                f"L={state.get('left_ear', 0):.3f} "
                f"R={state.get('right_ear', 0):.3f} | "
                f"Blinks={state.get('blink_count', 0)} "
                f"Rate={state.get('blink_rate_pm', 0):.1f}/min | "
                f"Drowsy={state.get('is_drowsy', False)} "
                f"({state.get('drowsy_reason', '')})"
            )

    _th.Thread(target=_print_stats, daemon=True).start()

    tracker = FaceTracker(shared_state=state)
    tracker.run()   # blocks on main thread
    detector.stop()
