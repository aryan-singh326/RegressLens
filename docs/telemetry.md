# Telemetry (opt-in)

Telemetry is **off** unless you turn it on:

* `rglns telemetry on [--endpoint URL]`, `rglns telemetry off`, `rglns telemetry status`;
* a one-time `[y/N]` prompt (default no), shown only in an interactive terminal
  and never in CI;
* `RGLNS_TELEMETRY=0` disables it unconditionally, even after opting in.

Even when it is on, nothing is sent unless an endpoint is configured
(`RGLNS_TELEMETRY_URL` or `--endpoint`). One event per command, with a
2-second timeout, and failures are ignored.

## What an event contains

Exactly these fields, and nothing else. The JSON Schema is published at
[`python/regresslens/telemetry_schema.json`](../python/regresslens/telemetry_schema.json)
(`additionalProperties: false`):

| field | example |
|---|---|
| schema_version | 1 |
| install_id | random UUID created when you opt in; not derived from your machine |
| command | `check` (one of the rglns commands, else `other`) |
| verdict_counts | `{"regression": 1, "equivalent": 14}` |
| python | `3.12.4` |
| numpy | `2.1.3` |
| platform | `Linux-x86_64` |
| regresslens | `0.2.0` |

Never sent: code, file paths, command arguments, labels, call-site names,
timings, hostnames.

`tests/test_telemetry.py` checks that with telemetry off no socket is ever
opened (`socket` is monkeypatched to raise), that events match the schema
exactly, and that labels, paths and script names never appear in a payload.
