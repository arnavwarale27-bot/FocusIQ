import time
import threading
from plyer import notification

class Notifier:
    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self._stop_event = threading.Event()
        
        # Cooldown dictionary mapping exact string keys out to unix time floats
        self._cooldowns = {
            "posture": 0.0,
            "phone": 0.0,
            "drowsy": 0.0,
            "break": 0.0,
            "focus": 0.0
        }
        
        # State timings mapping variables needing explicit continuous runtime checks
        self._bad_posture_start = None
        self._phone_detected_start = None
        self._low_focus_start = None
        self._session_start = time.time()
        
        self._last_badge_count = 0
        
    def start(self):
        """Optional explicit start hook mimicking daemon."""
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        
    def _can_notify(self, key: str, cooldown_duration: float) -> bool:
        """Returns True if the cooldown interval has officially passed for a specific event dict."""
        now = time.time()
        if now - self._cooldowns[key] >= cooldown_duration:
            self._cooldowns[key] = now
            return True
        return False

    def _notify(self, title: str, message: str, app_icon=None):
        """Execute the native plyer dispatch module strictly handling catching macOS permission errors silently."""
        try:
            notification.notify(
                title=title,
                message=message,
                app_name="AI Focus Monitor",
                timeout=5
            )
        except Exception as e:
            print(f"[Notifier] Failed to fire push notification: {e}")

    def run(self):
        print("[Notifier] Running (OSX Native Desktop Notifications).")
        # Initialize badge count gracefully immediately mapping state exactly before processing comparisons!
        self._last_badge_count = len(self.shared_state.get("badges", []))
        
        while not self._stop_event.is_set():
            now = time.time()
            state = self.shared_state
            
            # --- 1. Posture Check ---
            if state.get("bad_posture", False):
                if self._bad_posture_start is None:
                    self._bad_posture_start = now
                elif now - self._bad_posture_start >= 30.0:
                    if self._can_notify("posture", 120.0): # Cooldown: 2 minutes 
                        self._notify("⚠️ Fix Your Posture!", "You've been slouching for 30 seconds")
            else:
                self._bad_posture_start = None
                
            # --- 2. Phone Check ---
            if state.get("phone_detected", False):
                if self._phone_detected_start is None:
                    self._phone_detected_start = now
                elif now - self._phone_detected_start >= 10.0:
                    if self._can_notify("phone", 60.0): # Cooldown: 1 minute
                        self._notify("📵 Put Down Your Phone!", "Phone usage detected — stay focused!")
            else:
                self._phone_detected_start = None
                
            # --- 3. Drowsy Check ---
            if state.get("drowsy", False):
                if self._can_notify("drowsy", 180.0): # Cooldown: 3 minutes
                    self._notify("😴 Wake Up!", "You look drowsy — take a deep breath")
                    
            # --- 4. Break Reminder (20 minutes of runtime!) ---
            if now - self._session_start >= 1200.0:
                if self._can_notify("break", 1200.0): # Cooldown: 20 minutes
                    self._notify("☕ Take a Break!", "You've been focused for 20 minutes — rest your eyes for 20 seconds")
                    
            # --- 5. Low Focus Check ---
            score = state.get("focus_score", 100)
            if score < 30:
                if self._low_focus_start is None:
                    self._low_focus_start = now
                elif now - self._low_focus_start >= 60.0:
                    if self._can_notify("focus", 300.0): # Cooldown: 5 minutes 
                        self._notify("🎯 Refocus!", "Your focus score is dropping — get back on track")
            else:
                self._low_focus_start = None
                
            # --- 6. New Badge Checking (Ignoring `dashboard.py` widget wipe hooks completely) ---
            current_badges = state.get("badges", [])
            if len(current_badges) > self._last_badge_count:
                # Find all exact new badges explicitly added during this 5s delta window
                new_badges = current_badges[self._last_badge_count:]
                for badge in new_badges:
                    self._notify("🏆 Badge Earned!", f"You just earned: {badge}")
                self._last_badge_count = len(current_badges)

            # Throttle explicitly mapping sleep intervals perfectly safely against graceful exits! 
            self._stop_event.wait(5.0)
            
        print("[Notifier] Stopped.")

    def stop(self):
        self._stop_event.set()
