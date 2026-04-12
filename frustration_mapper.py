"""
frustration_mapper.py
Tracks behavioral signals indicative of frustration/distraction.

Signals:
1. Rapid head movement: yaw or pitch changing > 15° in < 1s
2. Long look-away: yaw > 20° for > 5s
3. Very low blink rate: < 3/min for > 30s
4. Drowsy + low focus: drowsy AND focus < 40
5. Repeated micro-looks: > 5 look-aways in 1 min
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
        self._frustration_score = 0.0

        self._head_history = collections.deque()
        self._look_away_start = None
        self._low_blink_start = None
        
        self._look_aways = collections.deque()
        self._was_looking_away = False

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

            active_signals = []

            # 1. Rapid head movement (>15 deg in <1s)
            self._head_history.append((now, yaw, pitch))
            while self._head_history and (now - self._head_history[0][0]) > 1.0:
                self._head_history.popleft()
                
            if len(self._head_history) > 1:
                yaws = [h[1] for h in self._head_history]
                pitches = [h[2] for h in self._head_history]
                if (max(yaws) - min(yaws)) > 15.0 or (max(pitches) - min(pitches)) > 15.0:
                    active_signals.append("Rapid head movement")

            # 2. Long look-away (yaw > 20 for > 5s)
            is_looking_away = yaw > 20.0
            if is_looking_away:
                if self._look_away_start is None:
                    self._look_away_start = now
                elif (now - self._look_away_start) > 5.0:
                    active_signals.append("Long look-away")
            else:
                self._look_away_start = None

            # 3. Very low blink rate (< 3 for > 30s)
            if blink_rpm < 3.0:
                if self._low_blink_start is None:
                    self._low_blink_start = now
                elif (now - self._low_blink_start) > 30.0:
                    active_signals.append("Very low blink rate")
            else:
                self._low_blink_start = None

            # 4. Drowsy + low focus (score < 40)
            if drowsy and focus_score < 40.0:
                active_signals.append("Drowsy + low focus")

            # 5. Repeated micro-looks (> 5 in 1 min)
            if is_looking_away and not self._was_looking_away:
                self._look_aways.append(now)
            self._was_looking_away = is_looking_away
            
            while self._look_aways and (now - self._look_aways[0]) > 60.0:
                self._look_aways.popleft()
                
            if len(self._look_aways) > 5:
                active_signals.append("Repeated micro-looks")

            accumulated_signals.update(active_signals)

            # Update EMA once per second
            if now - last_ema_time >= 1.0:
                # Each unique active signal contributes 35 points (clip max 100)
                # 0 signals = 0, 1 signal = 35 (Yellow), 2 signals = 70 (Red), 3+ signals = 100 (Red)
                target_score = len(accumulated_signals) * 35.0
                target_score = min(100.0, target_score)
                
                self._frustration_score = self._alpha * target_score + (1.0 - self._alpha) * self._frustration_score
                
                self.shared_state.update({
                    "frustration_score": round(self._frustration_score, 1),
                    "frustration_signals": list(accumulated_signals)
                })
                
                accumulated_signals.clear()
                last_ema_time = now

            time.sleep(interval)

        print("[Frustration] Stopped.")

if __name__ == "__main__":
    state = {}
    mapper = FrustrationMapper(shared_state=state)
    t = threading.Thread(target=mapper.run, daemon=True)
    t.start()
    time.sleep(5)
    mapper.stop()
