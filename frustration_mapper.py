"""
frustration_mapper.py — Phase 8 (Bonus)
Tracks eyebrow distance (landmarks 107 & 336) + blink velocity.
If eyebrow distance is decreasing AND blink velocity is high
→ tag as FRUSTRATION event in SQLite + export CSV.
"""

import threading
import time
import numpy as np
from scipy.spatial import distance as dist

from database import log_frustration

# Landmark indices for inner eyebrow corners
LEFT_EYEBROW_IDX  = 107
RIGHT_EYEBROW_IDX = 336

# Thresholds — tune these based on your face
BROW_SHRINK_THRESHOLD   = 5.0    # px decrease in distance per second
BLINK_VELOCITY_HIGH     = 0.08   # EAR change rate considered "fast blink"
FRUSTRATION_HOLD_FRAMES = 5      # consecutive frames matching both conditions


class FrustrationMapper:
    """
    Reads landmarks + blink_velocity from shared_state.
    Logs FRUSTRATION events to SQLite.

    shared_state keys read:
        landmarks      (list of (x,y,z) tuples)
        blink_velocity (float)

    shared_state keys written:
        frustration_detected (bool)
        frustration_count    (int)  — total events this session
    """

    def __init__(self, shared_state: dict, fps: float = 30.0):
        self.shared_state     = shared_state
        self.fps              = fps
        self._stop_event      = threading.Event()
        self._prev_brow_dist  : float | None = None
        self._consec_frames   = 0
        self._total_events    = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        interval = 1.0 / self.fps
        print("[Frustration] Running.")

        while not self._stop_event.is_set():
            landmarks = self.shared_state.get("landmarks", [])
            blink_vel = self.shared_state.get("blink_velocity", 0.0)

            if len(landmarks) < 468:
                time.sleep(interval)
                continue

            # ── Eyebrow distance ──────────────────────────────────────────
            left_brow  = np.array(landmarks[LEFT_EYEBROW_IDX][:2])
            right_brow = np.array(landmarks[RIGHT_EYEBROW_IDX][:2])
            brow_dist  = dist.euclidean(left_brow, right_brow)

            # ── Check for furrowing + fast blink ─────────────────────────
            is_furrowing = (
                self._prev_brow_dist is not None
                and (self._prev_brow_dist - brow_dist) > BROW_SHRINK_THRESHOLD
            )
            is_fast_blink = blink_vel > BLINK_VELOCITY_HIGH
            frustration = is_furrowing and is_fast_blink

            if frustration:
                self._consec_frames += 1
            else:
                self._consec_frames = 0

            # Log once we hit the hold threshold
            if self._consec_frames == FRUSTRATION_HOLD_FRAMES:
                self._total_events += 1
                print(f"[Frustration] 😤 FRUSTRATION event #{self._total_events} "
                      f"(brow_dist={brow_dist:.1f}, vel={blink_vel:.3f})")
                log_frustration(eyebrow_dist=brow_dist, blink_velocity=blink_vel)

            self.shared_state.update({
                "frustration_detected": frustration,
                "frustration_count"   : self._total_events,
                "brow_distance"       : round(brow_dist, 2),
            })

            self._prev_brow_dist = brow_dist
            time.sleep(interval)

        print("[Frustration] Stopped.")


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from database import init_db
    init_db()
    state = {"landmarks": [], "blink_velocity": 0.0}
    mapper = FrustrationMapper(shared_state=state)
    t = threading.Thread(target=mapper.run, daemon=True)
    t.start()
    time.sleep(5)
    mapper.stop()
