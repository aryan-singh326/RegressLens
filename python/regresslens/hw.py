"""Hardware description: CPU model, core count, cache sizes.

The fingerprint identifies the machine a session ran on, so reports can
flag baseline/current pairs measured on different hardware. Cache
sizes also give the size-regime breakpoints for the cost model
(stats/costmodel.py) and the buffer size for the memory canary.

Python and NumPy versions are deliberately NOT part of the fingerprint:
they are stored in their own session columns, because a NumPy upgrade
on the same machine is a software change, not a hardware one.
"""
import functools
import json
import os
import platform
import re
import subprocess
import sys

# Used when the platform does not report cache sizes.
_DEFAULT_CACHES = {"l1d": 32 * 1024, "l2": 1024 * 1024, "l3": 32 * 1024 * 1024}


def _parse_size(text):
    m = re.match(r"\s*(\d+)\s*([KMG]?)", text.upper())
    if not m:
        return None
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2)]
    return int(m.group(1)) * mult


def _linux_caches():
    base = "/sys/devices/system/cpu/cpu0/cache"
    out = {}
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return out
    for entry in entries:
        if not entry.startswith("index"):
            continue
        d = os.path.join(base, entry)
        try:
            with open(os.path.join(d, "level")) as f:
                level = int(f.read())
            with open(os.path.join(d, "type")) as f:
                ctype = f.read().strip()
            with open(os.path.join(d, "size")) as f:
                size = _parse_size(f.read())
        except (OSError, ValueError):
            continue
        if size is None or ctype == "Instruction":
            continue
        key = {1: "l1d", 2: "l2", 3: "l3"}.get(level)
        if key:
            out[key] = max(out.get(key, 0), size)
    return out


def _macos_caches():
    out = {}
    for key, name in (("l1d", "hw.l1dcachesize"), ("l2", "hw.l2cachesize"),
                      ("l3", "hw.l3cachesize")):
        try:
            value = subprocess.run(["sysctl", "-n", name], capture_output=True,
                                   text=True, timeout=5).stdout.strip()
            if value and int(value) > 0:
                out[key] = int(value)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return out


def _windows_caches():
    import ctypes
    from ctypes import wintypes

    class CacheDescriptor(ctypes.Structure):
        _fields_ = [("Level", ctypes.c_ubyte), ("Associativity", ctypes.c_ubyte),
                    ("LineSize", wintypes.WORD), ("Size", wintypes.DWORD),
                    ("Type", ctypes.c_int)]

    class _Union(ctypes.Union):
        _fields_ = [("Cache", CacheDescriptor), ("Reserved", ctypes.c_ulonglong * 2)]

    class Info(ctypes.Structure):
        _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_int),
                    ("u", _Union)]

    out = {}
    try:
        fn = ctypes.windll.kernel32.GetLogicalProcessorInformation
    except AttributeError:
        return out
    needed = wintypes.DWORD(0)
    fn(None, ctypes.byref(needed))
    count = needed.value // ctypes.sizeof(Info)
    if count == 0:
        return out
    buf = (Info * count)()
    if not fn(buf, ctypes.byref(needed)):
        return out
    for item in buf:
        if item.Relationship != 2:  # RelationCache
            continue
        c = item.u.Cache
        if c.Type == 1:  # CacheInstruction
            continue
        key = {1: "l1d", 2: "l2", 3: "l3"}.get(c.Level)
        if key:
            out[key] = max(out.get(key, 0), int(c.Size))
    return out


@functools.lru_cache(maxsize=1)
def cache_sizes():
    """{"l1d", "l2", "l3"} in bytes, with defaults for anything the
    platform does not report."""
    try:
        if sys.platform.startswith("linux"):
            found = _linux_caches()
        elif sys.platform == "darwin":
            found = _macos_caches()
        elif sys.platform == "win32":
            found = _windows_caches()
        else:
            found = {}
    except Exception:  # detection must never break a run
        found = {}
    caches = dict(_DEFAULT_CACHES)
    caches.update({k: v for k, v in found.items() if v})
    caches["detected"] = sorted(found)
    return caches


@functools.lru_cache(maxsize=1)
def cpu_model():
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith(("model name", "Model", "Hardware")):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout
            if out.strip():
                return out.strip()
        elif system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


@functools.lru_cache(maxsize=1)
def fingerprint_dict():
    caches = cache_sizes()
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "l1d": caches["l1d"],
        "l2": caches["l2"],
        "l3": caches["l3"],
    }


def fingerprint():
    """Stable JSON string for the sessions.hw_fingerprint column."""
    return json.dumps(fingerprint_dict(), sort_keys=True)


def describe(fp_json):
    """One-line human-readable form of a stored fingerprint."""
    try:
        d = json.loads(fp_json)
    except (TypeError, ValueError):
        return str(fp_json)
    return f"{d.get('cpu_model', '?')} ({d.get('cpu_count', '?')} threads, {d.get('system', '?')})"
