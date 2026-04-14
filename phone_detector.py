import time
import threading
from ultralytics import YOLO

class PhoneDetector:
    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self._stop_event = threading.Event()
        self.model = YOLO('yolov8n.pt')
        
        # EMA State for behavior tracking
        self._smoothed_score = 0.0
        self._alpha = 0.15
        self._behavior_active_since = None

    def start(self):
        """Optional API to auto-bind to a thread."""
        t = threading.Thread(target=self.run, daemon=True)
        t.start()

    def run(self):
        print("[PhoneDetector] Running (YOLOv8n + Behaviors).")
        while not self._stop_event.is_set():
            now = time.time()
            frame = self.shared_state.get("frame")
            if frame is None:
                time.sleep(0.1)
                continue

            # ── Method 1 - YOLO inference ──────────────────────────────────
            results = self.model(frame, verbose=False, classes=[67])
            visual_detected = False
            best_conf = 0.0

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        visual_detected = True
                        if conf > best_conf:
                            best_conf = conf

            # ── Method 2 - Behavioral scoring (more sensitive than YOLO) ──
            pitch = self.shared_state.get("pitch", 0.0)
            yaw = abs(self.shared_state.get("yaw", 0.0))
            blink_rate = self.shared_state.get("blink_rate_pm", 15.0)
            landmarks = self.shared_state.get("landmarks", [])
            
            target_score = 0.0
            
            # Head down continuously (lowered from 25° to 15°)
            if pitch > 15.0:
                target_score += 45.0
                
            # NEW: Looking straight ahead but head tilted down slightly
            # This catches the common case of phone held below screen level
            if yaw < 10.0 and 15.0 < pitch < 35.0:
                target_score += 35.0

            # Looking away + head down
            if yaw > 20.0 and pitch > 10.0:
                target_score += 30.0
                
            # Face drifting toward bottom of frame -> assume reading a phone embedded in lap
            if landmarks and len(landmarks) > 0:
                h, w = frame.shape[:2]
                nose_y = landmarks[0][1] # nose tip y-axis
                if nose_y > h * 0.70: 
                    target_score += 20.0
                    
            # Low blink + head down -> likely staring hard
            if blink_rate < 3.0 and pitch > 15.0:
                target_score += 15.0
                
            # Cap the arithmetic target mapping
            target_score = min(100.0, target_score)
            
            # EMA Smoothing
            self._smoothed_score = (self._alpha * target_score) + ((1.0 - self._alpha) * self._smoothed_score)
            
            # Check 2-second continuous trigger condition (lowered from 3s)
            behavioral_detected = False
            if self._smoothed_score >= 40.0:  # lowered from 50
                if self._behavior_active_since is None:
                    self._behavior_active_since = now
                elif now - self._behavior_active_since >= 2.0:  # lowered from 3s
                    behavioral_detected = True
            else:
                self._behavior_active_since = None

            # ── Decision Combination ───────────────────────────────────────
            phone_detected = visual_detected or behavioral_detected
            
            if phone_detected:
                method = "both" if (visual_detected and behavioral_detected) else ("visual" if visual_detected else "behavioral")
                conf_out = max(best_conf, self._smoothed_score / 100.0)
                print(f"[PhoneDetector] 📱 Phone detected! method={method} confidence={conf_out:.2f}")
                self.shared_state["phone_detected"] = True
                self.shared_state["phone_confidence"] = conf_out
                self.shared_state["phone_detection_method"] = method
            else:
                self.shared_state["phone_detected"] = False
                self.shared_state["phone_confidence"] = 0.0
                self.shared_state["phone_detection_method"] = "none"

            # Throttle to 500ms
            time.sleep(0.5)
            
        print("[PhoneDetector] Stopped.")

    def stop(self):
        self._stop_event.set()
