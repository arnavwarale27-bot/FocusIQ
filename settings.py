import os
import json

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")

DEFAULT_SETTINGS = {
    "ear_threshold": 0.20,
    "yaw_threshold": 20.0,
    "pitch_threshold": 20.0,
    "posture_angle": 20.0,
    "drowsy_seconds": 3.0,
    "blink_rate_min": 5.0,
    "blink_rate_max": 40.0,
}

class Settings:
    def __init__(self):
        self._data = DEFAULT_SETTINGS.copy()
        self.load()

    def load(self):
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r") as f:
                    loaded = json.load(f)
                    for k, v in loaded.items():
                        if k in self._data:
                            self._data[k] = float(v)
            except Exception as e:
                print(f"[Settings] Error loading settings.json: {e}")

    def save(self):
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump(self._data, f, indent=4)
        except Exception as e:
            print(f"[Settings] Error saving settings.json: {e}")

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value):
        if key in self._data:
            self._data[key] = float(value)
            self.save()  # Automatically save changes

    def reset_to_defaults(self):
        self._data = DEFAULT_SETTINGS.copy()
        self.save()

    def get_all(self):
        return self._data.copy()
