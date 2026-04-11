"""
head_pose.py — Phase 3
Uses 6 FaceMesh landmarks + cv2.solvePnP to compute Yaw, Pitch, Roll.
Flags 'looked_away' if |Yaw| > 30° or |Pitch| > 20°.
"""

import cv2
import numpy as np
import time
import threading

# ── The 6 landmark indices used for head pose ────────────────────────────────
#   These are key 3-D "anchor" points that are stable across facial expressions
POSE_LANDMARK_IDS = {
    "nose_tip"        : 1,
    "chin"            : 152,
    "left_eye_corner" : 33,
    "right_eye_corner": 263,
    "left_mouth"      : 61,
    "right_mouth"     : 291,
}

# ── "Model" 3-D coordinates of those same 6 points on a generic face ─────────
#   Units are arbitrary (mm-like). These come from a standard face model.
MODEL_POINTS_3D = np.array([
    (0.0,      0.0,      0.0),    # Nose tip
    (0.0,    -330.0,    -65.0),   # Chin
    (-225.0,  170.0,   -135.0),   # Left eye left corner
    ( 225.0,  170.0,   -135.0),   # Right eye right corner
    (-150.0, -150.0,   -125.0),   # Left mouth corner
    ( 150.0, -150.0,   -125.0),   # Right mouth corner
], dtype=np.float64)

# Thresholds (degrees)
YAW_THRESHOLD   = 30.0
PITCH_THRESHOLD = 20.0


def _build_camera_matrix(frame_width: int, frame_height: int) -> np.ndarray:
    """
    Approximate intrinsic camera matrix.
    focal_length ≈ frame width (a common approximation).
    """
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    return np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1        ],
    ], dtype=np.float64)


def _rotation_vector_to_euler(rot_vec: np.ndarray) -> tuple[float, float, float]:
    """
    Convert OpenCV rotation vector → Euler angles (Yaw, Pitch, Roll) in degrees.
    Steps:
      1. rot_vec (3x1) → 3x3 rotation matrix via cv2.Rodrigues
      2. Decompose rotation matrix → (Pitch, Yaw, Roll) using atan2
    """
    rot_mat, _ = cv2.Rodrigues(rot_vec)

    # Decompose the rotation matrix to Euler angles
    # Pitch = arcsin(-R[2,0])
    # Yaw   = arctan2(R[1,0], R[0,0])
    # Roll  = arctan2(R[2,1], R[2,2])
    pitch = np.degrees(np.arcsin(-rot_mat[2, 0]))
    yaw   = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
    roll  = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
    return yaw, pitch, roll


class HeadPoseEstimator:
    """
    Reads landmarks + frame from shared_state, computes Euler angles,
    and writes results back.

    shared_state keys written:
        yaw           (float)  degrees, positive = turned right
        pitch         (float)  degrees, positive = looking down
        roll          (float)  degrees
        looked_away   (bool)
    """

    def __init__(self, shared_state: dict, fps: float = 30.0):
        self.shared_state = shared_state
        self.fps          = fps
        self._stop_event  = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        interval = 1.0 / self.fps
        print("[HeadPose] Running.")

        while not self._stop_event.is_set():
            landmarks = self.shared_state.get("landmarks", [])
            frame     = self.shared_state.get("frame")

            if len(landmarks) < 468 or frame is None:
                self.shared_state.update({
                    "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "looked_away": False
                })
                time.sleep(interval)
                continue

            h, w = frame.shape[:2]
            cam_matrix    = _build_camera_matrix(w, h)
            dist_coeffs   = np.zeros((4, 1), dtype=np.float64)  # no lens distortion assumed

            # ── Extract the 6 image (2-D) points from landmarks ──────────
            ids = list(POSE_LANDMARK_IDS.values())
            image_points_2d = np.array(
                [(landmarks[i][0], landmarks[i][1]) for i in ids],
                dtype=np.float64,
            )

            # ── Solve PnP ─────────────────────────────────────────────────
            success, rot_vec, trans_vec = cv2.solvePnP(
                MODEL_POINTS_3D,
                image_points_2d,
                cam_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                time.sleep(interval)
                continue

            yaw, pitch, roll = _rotation_vector_to_euler(rot_vec)
            looked_away = abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD

            if looked_away:
                direction = []
                if abs(yaw) > YAW_THRESHOLD:
                    direction.append("left" if yaw < 0 else "right")
                if abs(pitch) > PITCH_THRESHOLD:
                    direction.append("down" if pitch > 0 else "up")
                print(f"[HeadPose] 👀 Looked away — "
                      f"Yaw={yaw:.1f}° Pitch={pitch:.1f}° ({', '.join(direction)})")

            self.shared_state.update({
                "yaw"        : round(yaw,   2),
                "pitch"      : round(pitch, 2),
                "roll"       : round(roll,  2),
                "looked_away": looked_away,
            })

            time.sleep(interval)

        print("[HeadPose] Stopped.")


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os, threading
    sys.path.insert(0, os.path.dirname(__file__))
    from face_tracker import FaceTracker

    state   = {}
    tracker = FaceTracker(shared_state=state)
    pose    = HeadPoseEstimator(shared_state=state)

    t = threading.Thread(target=pose.run, daemon=True)
    t.start()

    tracker.run()
    pose.stop()
