import threading
import time
import psutil

class ThermalManager:
    """
    Monitors CPU usage using psutil and updates shared_state['thermal_mode'].
    Modes: normal, eco, rest.
    """
    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self._stop_event = threading.Event()
        self.shared_state["thermal_mode"] = "normal"
        self.below_60_count = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        print("  ✅ ThermalManager started.")
        while not self._stop_event.is_set():
            cpu_percent = psutil.cpu_percent(interval=1.0)
            
            if cpu_percent > 85:
                if self.shared_state["thermal_mode"] != "rest":
                    self.shared_state["thermal_mode"] = "rest"
                    print("[Thermal] Rest mode — cooling down")
                self.below_60_count = 0
            elif cpu_percent > 70:
                if self.shared_state["thermal_mode"] != "eco":
                    self.shared_state["thermal_mode"] = "eco"
                    print("[Thermal] Eco mode activated")
                self.below_60_count = 0
            else:
                if cpu_percent < 60:
                    self.below_60_count += 3 # Actually interval is 10s outside of this, but we'll do 3 checks = 30s
                else:
                    self.below_60_count = 0
                    
                if self.below_60_count >= 3:
                    if self.shared_state["thermal_mode"] != "normal":
                        self.shared_state["thermal_mode"] = "normal"
                        print("[Thermal] Normal mode restored")
            
            # Wait for 10 seconds, but check stop event
            count = 0
            while count < 9 and not self._stop_event.is_set():
                time.sleep(1)
                count += 1
