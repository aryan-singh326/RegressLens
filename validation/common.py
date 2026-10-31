"""Shared helpers for the validation scripts."""
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(HERE, "..", "python"))


def save(name, payload):
    """Writes results/<name>.json with the machine and timestamp."""
    from regresslens import hw
    os.makedirs(RESULTS, exist_ok=True)
    payload = dict(payload)
    payload["_meta"] = {"when": time.strftime("%Y-%m-%d %H:%M"), "python": platform.python_version(),
                        "machine": hw.describe(hw.fingerprint())}
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"wrote {path}")
    return path


def progress(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
