"""Per-minute activity and an end-of-job efficiency summary, from vLLM's Prometheus /metrics.

The watchdog samples vLLM once a minute. Counting only the requests in flight at that instant misses
every request that starts and finishes between two samples, so a server answering short requests all
the time reads as idle and is shut down while in use. vLLM's cumulative counters do not have that
blind spot: if `vllm:request_success_total` or the token counters grew since the last sample, work
happened in that minute, however briefly.

Pure functions only (no I/O), so this is testable without a server:

    cur = parse_metrics(text)              # {metric name: value summed over label sets}
    d = delta(prev, cur)                   # what happened since the previous sample
    busy = is_busy(d, cur)                 # in flight now, or anything completed/generated
    stats.add_minute(d, busy)              # accumulate
    stats.summary(...)                     # one line for the log at shutdown
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Cumulative counters (monotonic while the server lives). The *_sum of a histogram is cumulative too:
# e2e request latency summed over finished requests is "request-seconds served".
COUNTERS = (
    "vllm:request_success_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:e2e_request_latency_seconds_sum",
)
# Instantaneous gauges.
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


def _canonical(name: str) -> str:
    # Older vLLM builds export `vllm_num_requests_running`; normalise to the `vllm:` spelling.
    return "vllm:" + name[len("vllm_") :] if name.startswith("vllm_") else name


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each metric over its label sets. Comments, blank lines and unparsable values are skipped."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_part, _, rest = line.partition(" ")
        if not rest:
            continue
        name = _canonical(name_part.split("{", 1)[0])
        if name not in COUNTERS and name not in GAUGES:
            continue
        try:
            value = float(rest.split()[0])
        except (ValueError, IndexError):
            continue
        out[name] = out.get(name, 0.0) + value
    return out


def delta(prev: dict[str, float] | None, cur: dict[str, float]) -> dict[str, float]:
    """Counter growth since `prev`. A counter that went DOWN means vLLM restarted, so its current value
    is all growth since then. With no previous sample there is nothing to compare: all zero."""
    if prev is None:
        return {k: 0.0 for k in COUNTERS}
    out = {}
    for k in COUNTERS:
        before, now = prev.get(k, 0.0), cur.get(k, 0.0)
        out[k] = now - before if now >= before else now
    return out


def in_flight(cur: dict[str, float]) -> int:
    return int(sum(cur.get(k, 0.0) for k in GAUGES))


def is_busy(d: dict[str, float], cur: dict[str, float]) -> bool:
    """Busy in this interval: a request is in flight now, or any request finished or any token was
    processed since the last sample."""
    return in_flight(cur) > 0 or any(d.get(k, 0.0) > 0 for k in COUNTERS)


def minute_line(d: dict[str, float], cur: dict[str, float]) -> str:
    return (
        f"[Metrics] reqs=+{int(d['vllm:request_success_total'])} "
        f"prompt_tok=+{int(d['vllm:prompt_tokens_total'])} "
        f"gen_tok=+{int(d['vllm:generation_tokens_total'])} "
        f"req_s=+{d['vllm:e2e_request_latency_seconds_sum']:.1f} "
        f"in_flight={in_flight(cur)}"
    )


@dataclass
class JobStats:
    minutes: int = 0
    busy_minutes: int = 0
    totals: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in COUNTERS})

    def add_minute(self, d: dict[str, float], busy: bool) -> None:
        self.minutes += 1
        self.busy_minutes += int(busy)
        for k in COUNTERS:
            self.totals[k] += d.get(k, 0.0)

    def summary(self, job_start: float, ready_at: float | None, end: float) -> str:
        """One line: where the allocation's time went. `ready_at` is when the endpoint was published
        (None if it never was). Utilisation is request-seconds per second of serving, so it can exceed
        1 with concurrent requests; busy minutes is the share of watched minutes with any activity."""
        total_min = (end - job_start) / 60
        if ready_at is None:
            return f"[Summary] allocation={total_min:.1f}min; the endpoint never became ready"
        startup_min = (ready_at - job_start) / 60
        serving_s = max(end - ready_at, 1e-9)
        req_s = self.totals["vllm:e2e_request_latency_seconds_sum"]
        busy_pct = 100 * self.busy_minutes / self.minutes if self.minutes else 0.0
        return (
            f"[Summary] allocation={total_min:.1f}min startup={startup_min:.1f}min "
            f"serving={serving_s / 60:.1f}min busy_minutes={self.busy_minutes}/{self.minutes} ({busy_pct:.0f}%) "
            f"requests={int(self.totals['vllm:request_success_total'])} "
            f"prompt_tok={int(self.totals['vllm:prompt_tokens_total'])} "
            f"gen_tok={int(self.totals['vllm:generation_tokens_total'])} "
            f"request_s={req_s:.0f} utilisation={req_s / serving_s:.1%} of serving, "
            f"{req_s / max(end - job_start, 1e-9):.1%} of allocation"
        )
