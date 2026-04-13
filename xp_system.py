import os
import time
import json
import threading

XP_FILE = os.path.join(os.path.dirname(__file__), "xp_data.json")

def get_level_threshold(xp):
    if xp < 100:
        return 1, 100
    if xp < 250:
        return 2, 250
    if xp < 500:
        return 3, 500
    if xp < 1000:
        return 4, 1000
    return 5, 1000

class XPSystem:
    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self._stop_event = threading.Event()
        
        self.xp = 0
        self.level = 1
        self.earned_badges = []
        self._load_data()
        
        self._last_xp_time = time.time()
        self._last_badge_time = time.time()
        self._session_start = time.time()
        
        # State tracking for badges
        self._focus_streak_start = time.time()
        self._focus_above_90_since = time.time()
        
        # Full session invalidators
        self._phone_detected_this_session = False
        self._bad_posture_this_session = False
        self._abnormal_blink_this_session = False
        
        # Continuous bad posture tracking (-5 XP per 2 mins)
        self._bad_posture_since = None
        self._bad_posture_penalized_intervals = 0
        
        # Reset any leftover badge flags
        self.shared_state["new_badge_trigger"] = None
        self._update_shared_state()

    def _load_data(self):
        if os.path.exists(XP_FILE):
            try:
                with open(XP_FILE, "r") as f:
                    data = json.load(f)
                    self.xp = data.get("xp", 0)
                    self.level = data.get("level", 1)
                    self.earned_badges = data.get("badges", [])
            except Exception as e:
                print(f"[XPSystem] Error loading DB: {e}")

    def _save_data(self):
        try:
            with open(XP_FILE, "w") as f:
                json.dump({
                    "xp": self.xp,
                    "level": self.level,
                    "badges": self.earned_badges
                }, f, indent=4)
        except Exception as e:
            print(f"[XPSystem] Error saving DB: {e}")

    def _award_badge(self, badge_str: str):
        if badge_str not in self.earned_badges:
            self.earned_badges.append(badge_str)
            self._save_data()
            print(f"[XPSystem] 🏆 NEW BADGE EARNED: {badge_str}")
            self.shared_state["new_badge_trigger"] = badge_str
            self._update_shared_state()

    def _add_xp(self, amount: int):
        self.xp = max(0, self.xp + amount)
        old_level = self.level
        
        self.level, next_threshold = get_level_threshold(self.xp)
        
        if self.level > old_level:
            self._award_badge("💪 Level Up")
            
        self._save_data()
        self._update_shared_state()

    def _update_shared_state(self):
        _, next_thresh = get_level_threshold(self.xp)
        self.shared_state["xp"] = self.xp
        self.shared_state["level"] = self.level
        self.shared_state["xp_to_next_level"] = next_thresh
        self.shared_state["badges"] = self.earned_badges.copy()

    def run(self):
        print("[XPSystem] Running.")
        
        # First Focus badge immediately granted for starting first session ever
        self._award_badge("🎯 First Focus")
        
        while not self._stop_event.is_set():
            now = time.time()
            state = self.shared_state
            
            score = state.get("focus_score", 100)
            phone = state.get("phone_detected", False)
            posture = state.get("bad_posture", False)
            drowsy = state.get("drowsy", False)
            blinks = state.get("blink_rate_pm", 15.0)
            
            blink_lo = state.get("blink_rate_min", 5.0)
            blink_hi = state.get("blink_rate_max", 40.0)
            
            # --- Invalidation flags for Session end badges ---
            if phone:
                self._phone_detected_this_session = True
            if posture:
                self._bad_posture_this_session = True
            if blinks < blink_lo or blinks > blink_hi:
                self._abnormal_blink_this_session = True
                
            # --- XP Loop (Every 60 Seconds) ---
            if now - self._last_xp_time >= 60.0:
                self._last_xp_time = now
                xp_delta = 0
                
                # Earning rules
                if score > 90:
                    xp_delta += 20
                elif score > 75:
                    xp_delta += 10
                    
                if not phone:
                    xp_delta += 5
                if not posture:
                    xp_delta += 3
                    
                if blink_lo <= blinks <= blink_hi:
                    xp_delta += 2
                    
                # Losing rules (Per minute deductions)
                if phone:
                    xp_delta -= 10
                if drowsy:
                    xp_delta -= 8
                    
                self._add_xp(xp_delta)

            # --- Extra posturing clock logic (-5 XP every 2 mins of continuous strict bad posture) ---
            if posture:
                if self._bad_posture_since is None:
                    self._bad_posture_since = now
                elif now - self._bad_posture_since >= 120.0 * (self._bad_posture_penalized_intervals + 1):
                    self._bad_posture_penalized_intervals += 1
                    self._add_xp(-5)
            else:
                self._bad_posture_since = None
                self._bad_posture_penalized_intervals = 0

            # --- Badge Check Loop (Every 30 Seconds) ---
            if now - self._last_badge_time >= 30.0:
                self._last_badge_time = now
                
                if score < 75:
                    self._focus_streak_start = now
                elif now - self._focus_streak_start >= 600: # 10 mins continuous > 75
                    self._award_badge("⚡ Focus Streak")
                    
                if score < 90:
                    self._focus_above_90_since = now
                elif now - self._focus_above_90_since >= 1800: # 30 mins continuous > 90
                    self._award_badge("🏆 Focus Champion")

            time.sleep(1.0)
            
        print("[XPSystem] Stopped.")

    def stop(self):
        # Shut down callback. Distribute "Session" badges if run time > 5 minutes (300s)
        session_time = time.time() - self._session_start
        if session_time >= 300.0:
            if not self._phone_detected_this_session:
                self._award_badge("📵 No Phone Zone")
            if not self._bad_posture_this_session:
                self._award_badge("🧘 Perfect Posture")
            if not self._abnormal_blink_this_session:
                self._award_badge("👁️ Blink Master")

        self._stop_event.set()
