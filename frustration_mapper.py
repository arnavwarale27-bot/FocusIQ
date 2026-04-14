"""
frustration_mapper.py
Tracks behavioral signals indicative of frustration/distraction.

Signals:
1. Rapid head movement: yaw or pitch changing > 25° in < 1s
2. Long look-away: yaw > 20° for > 8s
3. Very low blink rate: < 3/min for > 30s
4. Drowsy + low focus: drowsy AND focus < 40
5. Repeated micro-looks: > 5 look-aways in 1 min

Notes:
- Requires 10 consecutive frames with face detected before scoring.
- Poor lighting (frequent face detection drops) dampens score by 50%.
- Score decays immediately toward 0 when no signals are active.
"""

import threading
import time
import collections

class FrustrationMapper:
    """
    Reads state and computes an EMA-smoothed frustration score (0-100).
    """
    def __init__(self, shared_state: dict, fps: float = 30.0):
        self.shared_state = shared_state
        self.fps = fps
        self._stop_event = threading.Event()
        
        self._alpha = 0.1
        self._decay_alpha = 0.25          # faster decay when no signals
        self._frustration_score = 0.0

        self._head_history = collections.deque()
        self._look_away_start = None
        self._low_blink_start = None
        
        self._look_aways = collections.deque()
        self._was_looking_away = False

        # Stability guards
        self._consecutive_face_frames = 0
        self._face_detect_history = collections.deque(maxlen=60)  # 1 min window

    def stop(self):
        self._stop_event.set()

    def run(self):
        interval = 1.0 / self.fps
        print("[Frustration] Running.")
        
        last_ema_time = time.time()
        accumulated_signals = set()

        while not self._stop_event.is_set():
            now = time.time()
            
            # Read from state
            yaw = abs(self.shared_state.get("yaw", 0.0))
            pitch = abs(self.shared_state.get("pitch", 0.0))
            blink_rpm = self.shared_state.get("blink_rate_pm", 15.0)
            drowsy = self.shared_state.get("drowsy", False)
            focus_score = self.shared_state.get("focus_score", 100.0)
            face_detected = self.shared_state.get("face_detected", False)

            # ── Stability: track consecutive frames with face detected ──
            if face_detected:
                self._consecutive_face_frames += 1
            else:
                self._consecutive_face_frames = 0
            self._face_detect_history.append(face_detected)

            # Skip frustration scoring until face has been stable for 10 frames
            face_stable = self._consecutive_face_frames >= 10

            active_signals = []

            # 1. Rapid head movement (>17.5 deg in <1s) [Reduced 30%]
            self._head_history.append((now, yaw, pitch))
            while self._head_history and (now - self._head_history[0][0]) > 1.0:
                self._head_history.popleft()
                
            if face_stable and len(self._head_history) > 1:
                yaws = [h[1] for h in self._head_history]
                pitches = [h[2] for h in self._head_history]
                if (max(yaws) - min(yaws)) > 17.5 or (max(pitches) - min(pitches)) > 17.5:
                    active_signals.append("Rapid head movement")

            # 2. Long look-away (yaw > 14 for > 5.6s) [Reduced 30%]
            is_looking_away = yaw > 14.0
            if is_looking_away and face_stable:
                if self._look_away_start is None:
                    self._look_away_start = now
                elif (now - self._look_away_start) > 5.6:
                    active_signals.append("Long look-away")
            else:
                self._look_away_start = None

            # 3. Very low blink rate (< 2.1 for > 21s) [Reduced 30%]
            if blink_rpm < 2.1:
                if self._low_blink_start is None:
                    self._low_blink_start = now
                elif (now - self._low_blink_start) > 21.0:
                    active_signals.append("Very low blink rate")
            else:
                self._low_blink_start = None

            # 4. Drowsy + low focus (score < 28.0) [Reduced 30%]
            if face_stable and drowsy and focus_score < 28.0:
                active_signals.append("Drowsy + low focus")

            # 5. Repeated micro-looks (> 3 in 1 min) [Reduced 30%]
            if is_looking_away and not self._was_looking_away:
                self._look_aways.append(now)
            self._was_looking_away = is_looking_away
            
            while self._look_aways and (now - self._look_aways[0]) > 60.0:
                self._look_aways.popleft()
                
            if face_stable and len(self._look_aways) > 3:
                active_signals.append("Repeated micro-looks")

            accumulated_signals.update(active_signals)

            # Update EMA once per second
            if now - last_ema_time >= 1.0:
                # Each unique active signal contributes 35 points (clip max 100)
                target_score = len(accumulated_signals) * 35.0
                target_score = min(100.0, target_score)

                # ── Poor lighting dampening ──
                # If face detection drops frequently (>30% of last 60 frames), halve the score
                if len(self._face_detect_history) > 10:
                    detect_rate = sum(self._face_detect_history) / len(self._face_detect_history)
                    if detect_rate < 0.70:
                        target_score *= 0.5

                # Use faster decay alpha when target is 0 (no signals active)
                if target_score == 0.0:
                    alpha = self._decay_alpha   # aggressive decay toward 0
                else:
                    alpha = self._alpha

                self._frustration_score = alpha * target_score + (1.0 - alpha) * self._frustration_score
                
                self.shared_state.update({
                    "frustration_score": round(self._frustration_score, 1),
                    "frustration_signals": list(accumulated_signals)
                })
                
                accumulated_signals.clear()
                last_ema_time = now

            # ── Debug Print (Every 5 seconds) ──
            if int(now) % 5 == 0 and (now % 1) < interval * 1.5:
                # Ensure keys exist even if mapper logic was skipped
                f_score = self.shared_state.get("frustration_score", 0.0)
                f_signals = self.shared_state.get("frustration_signals", [])
                print(f"[Frustration DEBUG] Score: {f_score} | Signals: {f_signals}")

            time.sleep(interval)

        print("[Frustration] Stopped.")

if __name__ == "__main__":
    state = {}
    mapper = FrustrationMapper(shared_state=state)
    t = threading.Thread(target=mapper.run, daemon=True)
    t.start()
    time.sleep(5)
    mapper.stop()
