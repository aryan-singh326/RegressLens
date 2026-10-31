"""Opt-in, anonymous usage telemetry.

Off unless the user explicitly opts in: either `rglns telemetry on`, or
answering "y" to a one-time prompt shown only in an interactive
terminal (never in CI). `RGLNS_TELEMETRY=0` disables it unconditionally.

An event contains ONLY the fields in telemetry_schema.json (documented
in docs/telemetry.md): a random install id, the command name, verdict
counts, Python and NumPy versions, the platform, and the RegressLens
version. Never code, file paths, command arguments, call-site names or
timings.

Events are sent only when an endpoint is configured (RGLNS_TELEMETRY_URL
or `rglns telemetry on --endpoint URL`). Without one, nothing leaves the
machine even when telemetry is on.
"""
import json
import os
import platform
import sys
import uuid

SCHEMA_VERSION = 1
COMMANDS = ("run", "baseline", "check", "profile", "compare", "bisect", "watch", "verify",
            "sessions", "ci")
_CI_VARS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "JENKINS_URL", "TF_BUILD")


def config_path():
    return os.environ.get("RGLNS_CONFIG_PATH") or os.path.expanduser(
        os.path.join("~", ".regresslens", "config.json"))


def _load():
    try:
        with open(config_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(cfg):
    path = config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def state():
    """'on', 'off' or 'undecided' (undecided behaves as off)."""
    if os.environ.get("RGLNS_TELEMETRY") == "0":
        return "off"
    value = _load().get("telemetry")
    return value if value in ("on", "off") else "undecided"


def set_state(on, endpoint=None):
    cfg = _load()
    cfg["telemetry"] = "on" if on else "off"
    if on:
        cfg.setdefault("install_id", uuid.uuid4().hex)
        if endpoint:
            cfg["endpoint"] = endpoint
    _save(cfg)


def endpoint():
    return os.environ.get("RGLNS_TELEMETRY_URL") or _load().get("endpoint")


def status_text():
    s = state()
    lines = [f"telemetry: {s}" + (" (RGLNS_TELEMETRY=0)" if os.environ.get("RGLNS_TELEMETRY") == "0" else "")]
    lines.append(f"endpoint: {endpoint() or 'none configured (nothing is sent)'}")
    lines.append(f"config: {config_path()}")
    return "\n".join(lines)


def _interactive():
    if any(os.environ.get(v) for v in _CI_VARS):
        return False
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def maybe_prompt():
    """Asks once, in interactive terminals only. Default answer: no."""
    if state() != "undecided" or not _interactive():
        return
    print("RegressLens can send anonymous usage statistics (command name, verdict counts, "
          "Python/NumPy versions, platform; never code, paths or arguments). "
          "See docs/telemetry.md.", file=sys.stderr)
    # The prompt goes to stderr: stdout belongs to the measured program.
    sys.stderr.write("Enable telemetry? [y/N] ")
    sys.stderr.flush()
    try:
        answer = sys.stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        answer = ""
    set_state(answer in ("y", "yes"))


def build_event(command, verdict_counts=None):
    from . import __version__
    np_mod = sys.modules.get("numpy")
    return {
        "schema_version": SCHEMA_VERSION,
        "install_id": _load().get("install_id"),
        "command": command if command in COMMANDS else "other",
        "verdict_counts": {str(k): int(v) for k, v in (verdict_counts or {}).items()},
        "python": platform.python_version(),
        "numpy": getattr(np_mod, "__version__", None),
        "platform": f"{platform.system()}-{platform.machine()}",
        "regresslens": __version__,
    }


def record(command, verdict_counts=None):
    """Sends one event if telemetry is on and an endpoint is configured.
    Never raises and never delays a command by more than ~2 s."""
    if state() != "on":
        return False
    url = endpoint()
    if not url:
        return False
    try:
        import urllib.request
        body = json.dumps(build_event(command, verdict_counts)).encode()
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2).close()
        return True
    except Exception:
        return False
