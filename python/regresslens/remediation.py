"""
regresslens.remediation: cost-aware contiguity-fix recommendations.

Per the project brief: before suggesting np.ascontiguousarray(),
estimate the copy cost (array size x dtype bytes x calibrated copy
rate) and the expected downstream savings (accelerated runtime vs.
the numpy-fallback runtime actually observed). Report both numbers.
Recommend automatic application only when net benefit clears a
configurable safety margin. Default: recommend, never apply
automatically — that requires an explicit opt-in from the caller.

HONEST SCOPE NOTE: the project brief's "expected speedup x remaining
affected operations" formulation implies awareness of a pipeline's
downstream structure (how many later operations would also benefit
from fixing this one array's contiguity). v0.1 has no pipeline-graph
model — each rl.array call is independent from RegressLens's point of
view. This module estimates savings for the ONE call site actually
observed, not any downstream operations that might also be affected.
That's a real, smaller number than the brief's full formulation
describes, and it's reported as such rather than inflated with an
assumed multiplier.
"""
import numpy as np
import time


_CACHED_COPY_RATE_BYTES_PER_NS = None


def calibrate_copy_rate(sample_size=10_000_000, dtype=np.float64):
    """Measures np.ascontiguousarray()'s throughput on THIS machine,
    once, and caches it. This is the 'calibrated copy rate' the
    project brief specifies — measured on the actual hardware being
    used, not an assumed constant, since memory bandwidth varies
    significantly across machines.
    """
    global _CACHED_COPY_RATE_BYTES_PER_NS
    if _CACHED_COPY_RATE_BYTES_PER_NS is not None:
        return _CACHED_COPY_RATE_BYTES_PER_NS

    rng = np.random.default_rng(0)
    data = rng.uniform(-100, 100, sample_size * 2).astype(dtype)
    non_contiguous = data[::2]  # forces an actual copy below
    assert not non_contiguous.flags["C_CONTIGUOUS"]

    # A few reps, take the median — this calibration measurement
    # deserves the same "don't trust a single sample" discipline as
    # the kernel benchmarking does, even though it's a much smaller
    # concern here.
    samples = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        np.ascontiguousarray(non_contiguous)
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
    samples.sort()
    median_ns = samples[len(samples) // 2]

    total_bytes = non_contiguous.size * non_contiguous.itemsize
    rate = total_bytes / median_ns if median_ns > 0 else float("inf")
    _CACHED_COPY_RATE_BYTES_PER_NS = rate
    return rate


def estimate_copy_cost_ns(row_count, dtype_bytes):
    rate = calibrate_copy_rate()
    total_bytes = row_count * dtype_bytes
    return total_bytes / rate if rate > 0 else float("inf")


def estimate_remediation(row_count, dtype_bytes, fallback_runtime_samples,
                          accelerated_runtime_samples, safety_margin=2.0):
    """
    Returns a dict:
      copy_cost_ns: estimated one-time cost of np.ascontiguousarray()
      estimated_savings_ns: median(fallback) - median(accelerated),
        i.e. how much faster THIS call would be per invocation if
        contiguous. None if no accelerated reference data exists yet
        (this call site has never been observed running contiguous).
      net_benefit_ns: estimated_savings_ns - copy_cost_ns (single-call
        break-even; see recommend_apply for why this alone isn't the
        recommendation trigger)
      recommend_apply: True only if estimated_savings_ns exceeds
        copy_cost_ns by safety_margin, per the project brief's
        "benefit > 2x cost" default framing.
    """
    copy_cost_ns = estimate_copy_cost_ns(row_count, dtype_bytes)

    if not accelerated_runtime_samples:
        return {
            "copy_cost_ns": copy_cost_ns,
            "estimated_savings_ns": None,
            "net_benefit_ns": None,
            "recommend_apply": False,
            "reason": (
                "no historical accelerated (contiguous) runtime for this "
                "call shape yet — cannot estimate savings. Run this call "
                "site with a contiguous array at least once to establish "
                "a reference point."
            ),
        }

    fallback_sorted = sorted(fallback_runtime_samples)
    accel_sorted = sorted(accelerated_runtime_samples)
    median_fallback = fallback_sorted[len(fallback_sorted) // 2]
    median_accel = accel_sorted[len(accel_sorted) // 2]

    estimated_savings_ns = median_fallback - median_accel
    net_benefit_ns = estimated_savings_ns - copy_cost_ns

    recommend_apply = estimated_savings_ns > (copy_cost_ns * safety_margin)

    return {
        "copy_cost_ns": copy_cost_ns,
        "estimated_savings_ns": estimated_savings_ns,
        "net_benefit_ns": net_benefit_ns,
        "recommend_apply": recommend_apply,
        "reason": None,
    }
