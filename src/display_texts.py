import os
import json

display_texts_json_path = os.getenv("DISPLAY_TEXTS_JSON_PATH", "display_texts.json")

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = DotDict(v)
            elif isinstance(v, list):
                self[k] = [DotDict(i) if isinstance(i, dict) else i for i in v]


dt_data = {}
try:
    with open(display_texts_json_path, 'r', encoding='utf-8') as f:
        dt_data = json.load(f)
    dt = DotDict(dt_data)
except FileNotFoundError:
    raise Exception(f"FATAL: Display texts JSON file not found at '{display_texts_json_path}'. The application cannot start without it.")
except json.JSONDecodeError as e:
    raise Exception(f"FATAL: Error decoding display texts JSON file at '{display_texts_json_path}': {e}. The application cannot start.")
except Exception as e:
    raise Exception(f"FATAL: An unexpected error occurred while loading display texts from '{display_texts_json_path}': {e}. The application cannot start.")
