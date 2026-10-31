"""Opt-in telemetry (WP9)."""
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from regresslens import cli, telemetry

SCHEMA = json.load(open(os.path.join(os.path.dirname(telemetry.__file__),
                                     "telemetry_schema.json")))


class NetworkUsed(AssertionError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise NetworkUsed(f"network call attempted: {args[:2]}")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_off_by_env_makes_zero_network_calls(tmp_path, no_network, monkeypatch):
    monkeypatch.setenv("RGLNS_TELEMETRY", "0")
    monkeypatch.setenv("RGLNS_TELEMETRY_URL", "http://127.0.0.1:9/never")
    telemetry.set_state(True)  # even an explicit opt-in is overridden by the env var
    script = tmp_path / "s.py"
    script.write_text("import numpy as np\nnp.sum(np.ones(10))\n")
    assert cli.main(["sessions"]) == 0
    assert cli.main(["telemetry", "status"]) == 0
    assert telemetry.record("run", {"regression": 1}) is False


def test_undecided_is_off_and_never_prompts_non_interactively(no_network, monkeypatch):
    monkeypatch.delenv("RGLNS_TELEMETRY", raising=False)
    monkeypatch.setenv("RGLNS_TELEMETRY_URL", "http://127.0.0.1:9/never")
    telemetry.maybe_prompt()  # stdin is not a TTY under pytest
    assert telemetry.state() == "undecided"
    assert telemetry.record("check", {}) is False


def test_on_without_endpoint_sends_nothing(no_network, monkeypatch):
    monkeypatch.delenv("RGLNS_TELEMETRY", raising=False)
    monkeypatch.delenv("RGLNS_TELEMETRY_URL", raising=False)
    telemetry.set_state(True)
    assert telemetry.record("check", {"regression": 1}) is False


def test_event_matches_published_schema_exactly(monkeypatch):
    monkeypatch.delenv("RGLNS_TELEMETRY", raising=False)
    telemetry.set_state(True)
    event = telemetry.build_event("compare", {"regression": 1})
    assert set(event) == set(SCHEMA["properties"]) == set(SCHEMA["required"])
    assert event["command"] in SCHEMA["properties"]["command"]["enum"]
    assert len(event["install_id"]) == 32
    assert telemetry.build_event("rm -rf /secret/path", {})["command"] == "other"


def test_on_with_endpoint_posts_one_event_without_code_or_paths(tmp_path, monkeypatch):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.delenv("RGLNS_TELEMETRY", raising=False)
    monkeypatch.setenv("RGLNS_TELEMETRY_URL", f"http://127.0.0.1:{server.server_port}/e")
    telemetry.set_state(True)
    script = tmp_path / "secret_name.py"
    script.write_text("pass\n")
    assert cli.main(["run", "--label", "secret-label", "--", sys.executable, str(script)]) == 0
    server.shutdown()
    assert len(received) == 1
    payload = json.dumps(received[0])
    assert received[0]["command"] == "run"
    for leaked in ("secret_name", "secret-label", str(tmp_path)):
        assert leaked not in payload
