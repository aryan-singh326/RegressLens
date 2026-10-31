"""Observation of NumPy and rl.array calls.

Calls are aggregated in-process (aggregator.py) and written to the trace
database once, at exit or on an explicit `flush()`. `rglns run` adds
zero-code-change observation of plain NumPy through a backend installed
by the injected sitecustomize (bootstrap.py).
"""
import sys

from .. import callsite
from .aggregator import get as get_aggregator


def flush():
    """Writes all calls recorded so far in this process to the database."""
    get_aggregator().flush()


def caller_site(callee):
    """CallSite of the nearest user-code frame on the stack (the code that
    called into regresslens), noting its stack on first sight."""
    frame = callsite.first_user_frame(sys._getframe(1))
    if frame is None:
        return None
    site = callsite.resolve_frame(frame, callee=callee)
    agg = get_aggregator()
    if site.id not in agg.sites:
        agg.note_site(site, callsite.user_stack(frame))
    return site


__all__ = ["caller_site", "flush", "get_aggregator"]
