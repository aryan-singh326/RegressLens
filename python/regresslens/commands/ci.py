"""rglns ci comment: upsert the compare report as ONE pull-request comment.

Used by the GitHub Action (action.yml). The comment is identified by
the marker line `<!-- rglns-compare -->`: an existing comment with the
marker is edited in place, otherwise a new one is created, so a PR never
accumulates a comment per push.

Repository, PR number and API URL come from the standard GitHub Actions
environment (GITHUB_REPOSITORY, GITHUB_EVENT_PATH, GITHUB_API_URL) unless
given explicitly; the token from $GITHUB_TOKEN.
"""
import json
import os
import re
import urllib.error
import urllib.request

from .common import eprint
from .compare import MARKER


def _pr_number(explicit):
    if explicit:
        return int(explicit)
    event = os.environ.get("GITHUB_EVENT_PATH")
    if event and os.path.exists(event):
        with open(event, encoding="utf-8") as f:
            payload = json.load(f)
        number = (payload.get("pull_request") or {}).get("number") or payload.get("number")
        if number:
            return int(number)
    m = re.match(r"refs/pull/(\d+)/", os.environ.get("GITHUB_REF", ""))
    if m:
        return int(m.group(1))
    raise SystemExit("error: cannot determine the pull request number (use --pr)")


class GitHub:
    def __init__(self, api, repo, token):
        self.api, self.repo, self.token = api.rstrip("/"), repo, token

    def request(self, method, path, body=None):
        req = urllib.request.Request(
            self.api + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.token}",
                     "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json",
                     "User-Agent": "regresslens"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            return json.loads(data) if data else None

    def find_comment(self, pr):
        page = 1
        while True:
            batch = self.request("GET", f"/repos/{self.repo}/issues/{pr}/comments"
                                        f"?per_page=100&page={page}")
            for c in batch:
                if MARKER in (c.get("body") or ""):
                    return c
            if len(batch) < 100:
                return None
            page += 1

    def upsert(self, pr, body):
        existing = self.find_comment(pr)
        if existing:
            self.request("PATCH", f"/repos/{self.repo}/issues/comments/{existing['id']}",
                         {"body": body})
            return "updated", existing["id"]
        created = self.request("POST", f"/repos/{self.repo}/issues/{pr}/comments",
                               {"body": body})
        return "created", created["id"]


def run_ci_comment(args):
    with open(args.markdown, encoding="utf-8") as f:
        body = f.read()
    if MARKER not in body:
        body = MARKER + "\n" + body
    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"error: ${args.token_env} is not set")
    repo = args.repo or os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        raise SystemExit("error: cannot determine the repository (use --repo owner/name)")
    api = args.api or os.environ.get("GITHUB_API_URL", "https://api.github.com")
    gh = GitHub(api, repo, token)
    try:
        action, cid = gh.upsert(_pr_number(args.pr), body)
    except urllib.error.HTTPError as exc:
        eprint(f"error: GitHub API {exc.code}: {exc.read()[:300]!r}")
        return 2
    eprint(f"PR comment {action} (id {cid})")
    return 0
