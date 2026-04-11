"""
posture.py — Phase 7
Uses MediaPipe Pose to track shoulders (11, 12) and nose (0).
If neck-to-shoulder angle indicates forward lean > 15° for 3 minutes,
sends a desktop notification via plyer.
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from plyer import notification

# ── MediaPipe Pose setup ─────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Landmark indices
NOSE_IDX           = 0
LEFT_SHOULDER_IDX  = 11
RIGHT_SHOULDER_IDX = 12

# Thresholds
FORWARD_LEAN_THRESHOLD = 15.0   # degrees
ALERT_AFTER_SECONDS    = 180    # 3 minutes


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle at vertex B formed by rays BA and BC.
    a, b, c : 2-D numpy arrays of (x, y) positions.
    Returns angle in degrees.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


class PostureDetector:
    """
    Reads the face-tracker frame from shared_state (no second camera needed),
    runs MediaPipe Pose, and checks for forward lean.
    Sends a desktop notification after ALERT_AFTER_SECONDS of bad posture.

    shared_state keys read:
        frame             (ndarray)  provided by FaceTracker
    shared_state keys written:
        neck_angle        (float)  degrees of neck lean
        bad_posture       (bool)
        posture_alert_sent(bool)   resets when posture corrects
    """

    def __init__(self, shared_state: dict, camera_index: int = 0):
        # camera_index kept for API compatibility but is no longer used
        self.shared_state = shared_state
        self._stop_event  = threading.Event()
        self._bad_start   = None

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def stop(self):
        self._stop_event.set()

    def _send_alert(self):
        print("[Posture] 📢 Sending posture notification.")
        try:
            notification.notify(
                title="Posture Alert 🙆",
                message="You've been leaning forward for 3 minutes. Sit up straight!",
                app_name="AI Focus Monitor",
                timeout=10,
            )
        except Exception as e:
            print(f"[Posture] Notification error: {e}")

    def run(self):
        """
        Main loop — runs in a daemon thread.
        Reads the frame already captured by FaceTracker from shared_state
        so it does NOT open its own camera (avoids macOS threading conflict).
        """
        print("[Posture] Running (sharing FaceTracker frame).")

        while not self._stop_event.is_set():
            frame = self.shared_state.get("frame")
            if frame is None:
                # Wait for FaceTracker to produce a frame
                time.sleep(0.1)
                continue

            frame = frame.copy()    # don't mutate the shared frame
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            neck_angle  = 0.0
            bad_posture = False

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                nose       = np.array([lm[NOSE_IDX].x * w,           lm[NOSE_IDX].y * h])
                l_shoulder = np.array([lm[LEFT_SHOULDER_IDX].x * w,  lm[LEFT_SHOULDER_IDX].y * h])
                r_shoulder = np.array([lm[RIGHT_SHOULDER_IDX].x * w, lm[RIGHT_SHOULDER_IDX].y * h])

                mid_shoulder = (l_shoulder + r_shoulder) / 2.0
                ref_above    = mid_shoulder - np.array([0, 100])
                neck_angle   = _angle_degrees(ref_above, mid_shoulder, nose)
                bad_posture  = neck_angle > FORWARD_LEAN_THRESHOLD

                if bad_posture:
                    print(f"[Posture] ⚠️  Bad posture: {neck_angle:.1f}°")

            # ── Timer logic ───────────────────────────────────────────────
            now = time.time()
            if bad_posture:
                if self._bad_start is None:
                    self._bad_start = now
                elif now - self._bad_start >= ALERT_AFTER_SECONDS:
                    if not self.shared_state.get("posture_alert_sent", False):
                        self._send_alert()
                        self.shared_state["posture_alert_sent"] = True
            else:
                self._bad_start = None
                self.shared_state["posture_alert_sent"] = False

            self.shared_state.update({
                "neck_angle" : round(neck_angle, 2),
                "bad_posture": bad_posture,
            })

            # Pose is CPU-heavy; 10 fps is plenty for posture checking
            time.sleep(0.1)

        self.pose.close()
        print("[Posture] Stopped.")


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
    import threading as _threading

    state = {}
    cap   = cv2.VideoCapture(0)   # camera opened on the main thread (safe)
    detector = PostureDetector(shared_state=state)
    t = _threading.Thread(target=detector.run, daemon=True)
    t.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        state["frame"] = frame
        cv2.imshow("Posture (standalone)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()
