"""The GitHub Action's logic, end to end against a local fake GitHub API
(WP9): a regressing PR and a clean PR, each pushed twice. Asserts the PR
comment content, that one comment is upserted (not duplicated), and the
check status (the step's exit code). The same commands run in action.yml;
.github/workflows/action-e2e.yml exercises the published action on
GitHub itself."""
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from conftest import commit, make_git_repo, run_rglns

BENCH = "import numpy as np\nfrom signals import smooth\nx = np.random.default_rng(0).uniform(-1, 1, 300_000)\nfor _ in range(12):\n    smooth(x)\n"
SIGNALS = "import numpy as np\ndef smooth(x):\n    return np.convolve(x, np.ones({w}) / {w}, mode='valid')\n"


class FakeGitHub:
    def __init__(self):
        self.comments = {}  # id -> {"pr": n, "body": str}
        self.next_id = 100
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self):
                return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

            def do_GET(self):
                assert self.headers["Authorization"] == "Bearer test-token"
                pr = int(self.path.split("/issues/")[1].split("/")[0])
                self._json(200, [{"id": cid, "body": c["body"]}
                                 for cid, c in fake.comments.items() if c["pr"] == pr])

            def do_POST(self):
                pr = int(self.path.split("/issues/")[1].split("/")[0])
                cid = fake.next_id
                fake.next_id += 1
                fake.comments[cid] = {"pr": pr, "body": self._body()["body"]}
                self._json(201, {"id": cid})

            def do_PATCH(self):
                cid = int(self.path.rsplit("/", 1)[1])
                fake.comments[cid]["body"] = self._body()["body"]
                self._json(200, {"id": cid})

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.api = f"http://127.0.0.1:{self.server.server_port}"

    def for_pr(self, pr):
        return [c["body"] for c in self.comments.values() if c["pr"] == pr]


def action(repo, pr, api, tmp_path):
    """The action.yml steps: compare base..HEAD, upsert the comment, and
    fail the check only on a regression verdict."""
    md, js = tmp_path / f"pr{pr}.md", tmp_path / f"pr{pr}.json"
    cmp = run_rglns("compare", "main", "HEAD", "--cmd", "python bench.py", "--seed", "1",
                    "--max-blocks", "40", "--markdown", str(md), "--json", str(js), cwd=repo)
    post = run_rglns("ci", "comment", "--markdown", str(md), "--repo", "owner/repo",
                     "--pr", str(pr), "--api", api, cwd=repo,
                     env={"GITHUB_TOKEN": "test-token"})
    assert post.returncode == 0, post.stderr
    check_failed = cmp.returncode == 1
    return check_failed, json.loads(js.read_text())


@pytest.mark.slow
def test_action_end_to_end(tmp_path):
    repo = make_git_repo(str(tmp_path / "repo"), {"bench.py": BENCH,
                                                   "signals.py": SIGNALS.format(w=5)})
    subprocess.run(["git", "branch", "-M", "main"], cwd=repo, check=True)
    gh = FakeGitHub()

    subprocess.run(["git", "checkout", "-q", "-b", "slow-pr"], cwd=repo, check=True)
    commit(repo, {"signals.py": SIGNALS.format(w=60)}, "wider smoothing window")
    failed, result = action(repo, 1, gh.api, tmp_path)
    assert failed and result["verdict"] == "regression"
    [body] = gh.for_pr(1)
    assert body.startswith("<!-- rglns-compare -->") and "**regression**" in body
    assert "signals.py" in body  # per-call-site drill-down
    action(repo, 1, gh.api, tmp_path)  # a second push edits, never duplicates
    assert len(gh.for_pr(1)) == 1

    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "docs-pr"], cwd=repo, check=True)
    commit(repo, {"README.md": "docs only\n"}, "docs")
    failed, result = action(repo, 2, gh.api, tmp_path)
    assert not failed and result["verdict"] in ("equivalent", "inconclusive")
    [body] = gh.for_pr(2)
    assert "**regression**" not in body
    gh.server.shutdown()
