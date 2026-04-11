"""
face_tracker.py — Phase 2
Opens the webcam, detects 468 FaceMesh landmarks, and draws them live.
Shares detected landmarks into a shared_state dict for other modules.
"""

import cv2
import mediapipe as mp
import threading

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

# Drawing spec: small green dots on every landmark
LANDMARK_DRAWING_SPEC = mp_drawing.DrawingSpec(
    color=(0, 255, 0), thickness=1, circle_radius=1
)
CONNECTION_DRAWING_SPEC = mp_drawing.DrawingSpec(
    color=(0, 128, 255), thickness=1
)


class FaceTracker:
    """
    Captures frames from the webcam, runs MediaPipe FaceMesh,
    draws landmarks, and writes results to shared_state.
    """

    def __init__(self, shared_state: dict, camera_index: int = 0):
        self.shared_state = shared_state
        self.camera_index = camera_index
        self._stop_event = threading.Event()

        # FaceMesh: refine_landmarks=True gives more precise eye/iris points
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,   # enables 478-landmark model (iris)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def stop(self):
        """Signal the tracker to stop."""
        self._stop_event.set()

    def run(self):
        """Main loop — call this in a dedicated thread."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[FaceTracker] ERROR: Cannot open camera.")
            return

        print("[FaceTracker] Camera opened. Press Q in the window to quit.")

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[FaceTracker] Warning: dropped frame.")
                continue

            # MediaPipe works with RGB images
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False          # small perf boost
            results = self.face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # ── Store raw landmarks in shared state ──────────────────────
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                # Convert to plain list of (x, y, z) tuples (normalised 0–1)
                h, w, _ = frame.shape
                lm_list = [
                    (lm.x * w, lm.y * h, lm.z)
                    for lm in landmarks.landmark
                ]
                self.shared_state["landmarks"]    = lm_list
                self.shared_state["face_detected"] = True

                # Draw the mesh on the frame
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles
                        .get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles
                        .get_default_face_mesh_contours_style(),
                )
            else:
                self.shared_state["face_detected"] = False
                self.shared_state["landmarks"]     = []

            # ── Store the annotated frame for the dashboard ──────────────
            self.shared_state["frame"] = frame.copy()

            # NOTE: cv2.imshow is intentionally NOT called here.
            # On macOS, imshow must run on the main thread.
            # The PyQt5 dashboard (WebcamLabel) displays this frame.

        cap.release()
        self.face_mesh.close()
        print("[FaceTracker] Stopped.")



# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    state = {}
    tracker = FaceTracker(shared_state=state)
    tracker.run()   # blocking — runs in main thread for testing
