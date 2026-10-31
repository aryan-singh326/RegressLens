"""A minimal fake of the GitHub issue-comments REST API, for testing the
Action end to end without touching a real repository.

    python tools/fake_github.py --port 8765 --dump comments.json &

Implements GET/POST /repos/{owner}/{repo}/issues/{n}/comments and PATCH
/repos/{owner}/{repo}/issues/comments/{id}; every change is written to
--dump as {"<id>": {"pr": n, "body": "..."}}.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()
    comments = {}
    counter = [100]

    def save():
        with open(args.dump, "w") as f:
            json.dump(comments, f, indent=2)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _pr(self):
            return int(self.path.split("/issues/")[1].split("/")[0].split("?")[0])

        def do_GET(self):
            pr = self._pr()
            self._send(200, [{"id": int(cid), "body": c["body"]}
                             for cid, c in comments.items() if c["pr"] == pr])

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            cid = counter[0]
            counter[0] += 1
            comments[str(cid)] = {"pr": self._pr(), "body": body["body"]}
            save()
            self._send(201, {"id": cid})

        def do_PATCH(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            cid = self.path.rsplit("/", 1)[1]
            comments[cid]["body"] = body["body"]
            save()
            self._send(200, {"id": int(cid)})

        def log_message(self, *a):
            pass

    save()
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
