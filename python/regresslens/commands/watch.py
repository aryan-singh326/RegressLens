"""rglns watch: change-point detection over a recurring pipeline's history.

    rglns watch --label nightly [--k 8] [--threshold 0.9] [--per-call-site]

For daily pipelines each day's `rglns run --label nightly` is one session,
so nobody has to choose two points to compare. Each session becomes one
observation, the median over its runs of log(wall time) - log(canary
blend). BOCPD (stats/bocpd.py) then alerts when P(change within the last
k sessions) > threshold. Exit code 1 means the latest session is in an
alert state (for cron jobs and CI).
"""
import datetime
import json
import math
from collections import defaultdict

import numpy as np

from .. import analysis, store
from ..stats import bocpd
from .common import eprint, write_output


def _adjusted(run):
    t = math.log(run.wall_ns)
    if run.canary_compute_ns and run.canary_mem_ns:
        t -= 0.5 * (math.log(run.canary_compute_ns) + math.log(run.canary_mem_ns))
    return t


def session_series(sessions):
    xs, kept = [], []
    for s in sessions:
        vals = [_adjusted(r) for r in s.runs if r.wall_ns]
        if vals:
            xs.append(float(np.median(vals)))
            kept.append(s)
    return np.array(xs), kept


def site_series(sessions):
    """{site_id: (values, session indices)}: per-session median log time of
    each call site (canary-adjusted)."""
    out = defaultdict(lambda: ([], []))
    for idx, s in enumerate(sessions):
        per_site = defaultdict(list)
        for r in s.runs:
            blend = (0.5 * (math.log(r.canary_compute_ns) + math.log(r.canary_mem_ns))
                     if r.canary_compute_ns and r.canary_mem_ns else 0.0)
            by_site = defaultdict(list)
            for c in r.calls:
                by_site[c.call_site_id].append(c)
            for site_id, rows in by_site.items():
                v = analysis._run_value(rows, None)
                if v is not None:
                    per_site[site_id].append(v - blend)
        for site_id, vals in per_site.items():
            out[site_id][0].append(float(np.median(vals)))
            out[site_id][1].append(idx)
    return out


def run_watch(args):
    conn = store.connect()
    try:
        found = store.find_sessions(conn, label=args.label)
        sessions = store.load_sessions(conn, [s.session_id for s in found],
                                       with_calls=args.per_call_site)
        info = {}
        if args.per_call_site:
            ids = {c.call_site_id for s in sessions for r in s.runs for c in r.calls}
            info = store.call_site_info(conn, ids)
    finally:
        conn.close()
    xs, kept = session_series(sessions)
    if len(xs) < args.warmup + args.k + 1:
        eprint(f"need at least {args.warmup + args.k + 1} sessions labeled '{args.label}' "
               f"(have {len(xs)})")
        return 2
    alerts, probs = bocpd.detect(xs, k=args.k, threshold=args.threshold, hazard=args.hazard,
                                 warmup=args.warmup)
    result = {"label": args.label, "sessions": len(xs), "alerts": [], "sites": []}
    ref = float(np.median(xs[:args.warmup]))
    print(f"rglns watch '{args.label}': {len(xs)} sessions, BOCPD hazard {args.hazard}, "
          f"alert when P(change in last {args.k}) > {args.threshold}")
    print(f"  {'session':<12} {'date':<16} {'vs start':>9} {'P(change)':>9}")
    for i, (s, x) in enumerate(zip(kept, xs)):
        when = datetime.datetime.fromtimestamp(s.started_at).strftime("%Y-%m-%d %H:%M")
        p = "" if math.isnan(probs[i]) else f"{probs[i]:.2f}"
        flag = "  <-- ALERT" if any(a.index == i for a in alerts) else ""
        print(f"  {s.session_id[:12]:<12} {when:<16} {math.expm1(x - ref):+8.1%} {p:>9}{flag}")
    for a in alerts:
        before = xs[max(0, a.onset - 5):a.onset]
        after = xs[a.onset:a.index + 1]
        shift = math.expm1(np.median(after) - np.median(before)) if len(before) else float("nan")
        result["alerts"].append({"session": kept[a.index].session_id, "probability": a.probability,
                                 "onset_session": kept[a.onset].session_id, "shift": shift})
        print(f"change detected at session {kept[a.index].session_id[:12]} "
              f"(onset ~{kept[a.onset].session_id[:12]}), P = {a.probability:.2f}, "
              f"shift {shift:+.1%}")
    if args.per_call_site:
        for site_id, (vals, idxs) in site_series(sessions).items():
            if len(vals) < args.warmup + args.k + 1:
                continue
            s_alerts, _ = bocpd.detect(vals, k=args.k, threshold=args.threshold,
                                       hazard=args.hazard, warmup=args.warmup)
            if s_alerts:
                disp = analysis._site_display(info.get(site_id), site_id)
                result["sites"].append({"site": disp, "alerts": len(s_alerts)})
                print(f"  call site {disp}: {len(s_alerts)} change(s), latest at session "
                      f"{sessions[idxs[s_alerts[-1].index]].session_id[:12]}")
    latest_alert = not math.isnan(probs[-1]) and probs[-1] > args.threshold
    if args.json:
        write_output(args.json, json.dumps(result, indent=2, default=float))
    args.verdict_counts = {"alert": len(alerts)}
    return 1 if latest_alert else 0
