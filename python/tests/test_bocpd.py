"""Change-point detection for recurring pipelines (WP6)."""
import math
import os
import sys

import numpy as np

from regresslens import cli, store
from regresslens.stats import bocpd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "validation"))
import simlib  # noqa: E402


def test_false_alarms_at_most_one_per_500_null_sessions():
    assert simlib.bocpd_false_alarms(series_count=30, seed=1) <= 1.0


def test_detection_delay_for_10pct_shift_at_3pct_noise():
    delays = simlib.bocpd_delays(trials=150, seed=2)
    assert np.mean(delays <= 5) >= 0.9
    assert np.median(delays) <= 5


def test_no_alert_during_warmup_or_on_constant_series():
    alerts, probs = bocpd.detect(np.zeros(100) + 3.0)
    assert alerts == [] and np.isnan(probs[:10]).all()


def _record_history(label, values, runs=3):
    """Writes a recorded history: one session per value, `runs` runs each."""
    conn = store.connect()
    for day, v in enumerate(values):
        sid = store.create_session(conn, label, started_at=1.7e9 + day * 86400)
        for r in range(runs):
            store.upsert_run(conn, f"{sid}-{r}", sid, wall_ns=math.exp(v) * 1e9, exit_code=0,
                             started_at=1.7e9 + day * 86400 + r,
                             canary_samples=[{"compute_ns": 1e5, "mem_ns": 2e7}])
    conn.close()


def test_replay_injected_regression_alerts(capsys):
    rng = np.random.default_rng(3)
    history = list(rng.normal(0, 0.02, 40))
    history += list(rng.normal(math.log(1.15), 0.02, 4))  # regression injected at day 40
    _record_history("nightly", history)
    rc = cli.main(["watch", "--label", "nightly"])
    out = capsys.readouterr().out
    assert rc == 1  # the latest session is in an alert state
    assert "change detected" in out


def test_replay_null_history_is_quiet(capsys):
    _record_history("steady", list(np.random.default_rng(4).normal(0, 0.02, 45)))
    assert cli.main(["watch", "--label", "steady"]) == 0
    assert "change detected" not in capsys.readouterr().out
