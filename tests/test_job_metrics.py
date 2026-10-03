"""Offline tests for src/job_metrics.py: no server, no Slurm."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import job_metrics as jm


def sample(
    running=0, waiting=0, success=0, prompt=0, gen=0, e2e=0.0, legacy=False
) -> str:
    """A /metrics body in vLLM's format: labelled series, HELP/TYPE comments, other metrics."""
    m = "vllm_" if legacy else "vllm:"
    lab = '{engine="0",model_name="my-local-model"}'
    return "\n".join(
        [
            f"# HELP {m}num_requests_running Number of requests in model execution batches.",
            f"# TYPE {m}num_requests_running gauge",
            f"{m}num_requests_running{lab} {running}",
            f"{m}num_requests_waiting{lab} {waiting}",
            # success is labelled by finished_reason; the parser must sum the label sets
            f'vllm:request_success_total{{engine="0",finished_reason="stop"}} {success - 1 if success else 0}',
            f'vllm:request_success_total{{engine="0",finished_reason="length"}} {1 if success else 0}',
            f"vllm:prompt_tokens_total{lab} {prompt}",
            f"vllm:generation_tokens_total{lab} {gen}",
            f"vllm:e2e_request_latency_seconds_sum{lab} {e2e}",
            f"vllm:e2e_request_latency_seconds_count{lab} {success}",
            f"vllm:gpu_cache_usage_perc{lab} 0.01",
            "process_cpu_seconds_total 12.5",
            "",
        ]
    )


def test_parse_sums_label_sets_and_ignores_the_rest():
    cur = jm.parse_metrics(
        sample(running=2, waiting=1, success=5, prompt=100, gen=40, e2e=12.5)
    )
    assert cur["vllm:request_success_total"] == 5
    assert cur["vllm:prompt_tokens_total"] == 100
    assert cur["vllm:e2e_request_latency_seconds_sum"] == 12.5
    assert jm.in_flight(cur) == 3
    assert "process_cpu_seconds_total" not in cur
    assert "vllm:gpu_cache_usage_perc" not in cur


def test_legacy_underscore_names_are_normalised():
    cur = jm.parse_metrics(sample(running=1, legacy=True))
    assert jm.in_flight(cur) == 1


def test_first_sample_has_no_delta():
    cur = jm.parse_metrics(sample(success=3, prompt=50))
    assert all(v == 0 for v in jm.delta(None, cur).values())


def test_counter_reset_counts_the_new_value():
    prev = jm.parse_metrics(sample(success=10, prompt=1000))
    cur = jm.parse_metrics(sample(success=2, prompt=40))  # vLLM restarted
    d = jm.delta(prev, cur)
    assert d["vllm:request_success_total"] == 2
    assert d["vllm:prompt_tokens_total"] == 40


def test_a_request_between_samples_is_busy():
    """The watchdog's old blind spot: nothing in flight at either sample, yet work happened."""
    prev = jm.parse_metrics(sample(success=4, prompt=300, gen=50, e2e=9.0))
    cur = jm.parse_metrics(sample(success=5, prompt=1200, gen=500, e2e=29.0))
    d = jm.delta(prev, cur)
    assert jm.in_flight(cur) == 0
    assert jm.is_busy(d, cur)


def test_nothing_happened_is_idle_and_in_flight_is_busy():
    prev = jm.parse_metrics(sample(success=5, prompt=1200))
    same = jm.parse_metrics(sample(success=5, prompt=1200))
    assert not jm.is_busy(jm.delta(prev, same), same)
    long_running = jm.parse_metrics(sample(running=1, success=5, prompt=1200))
    assert jm.is_busy(jm.delta(prev, long_running), long_running)


def test_minute_line_reports_growth():
    prev = jm.parse_metrics(sample(success=4, prompt=300, gen=50, e2e=9.0))
    cur = jm.parse_metrics(sample(running=1, success=5, prompt=1200, gen=500, e2e=29.0))
    line = jm.minute_line(jm.delta(prev, cur), cur)
    assert (
        line == "[Metrics] reqs=+1 prompt_tok=+900 gen_tok=+450 req_s=+20.0 in_flight=1"
    )


def test_replay_of_a_job_the_old_watchdog_called_idle():
    """Shaped like job 7661920: requests of a few seconds each, never in flight at a sample.
    The old rule saw 15 idle minutes and shut the job down; the counters see the activity."""
    counters = [  # (success, prompt, gen, e2e) at each one-minute sample
        (0, 0, 0, 0.0),
        (0, 0, 0, 0.0),
        (0, 0, 0, 0.0),
        (1, 10, 13, 2.0),
        (2, 20, 25, 4.0),
        (2, 20, 25, 4.0),
        (3, 300, 51, 7.0),
        (4, 330, 63, 9.0),
        (4, 330, 63, 9.0),
        (4, 330, 63, 9.0),
        (5, 340, 85, 11.0),
        (5, 340, 85, 11.0),
        (6, 363, 2320, 31.0),
        (6, 363, 2320, 31.0),
        (6, 363, 2320, 31.0),
        (7, 769, 2322, 33.0),
    ]
    stats, prev = jm.JobStats(), None
    old_rule_busy = 0
    for success, prompt, gen, e2e in counters:
        cur = jm.parse_metrics(sample(success=success, prompt=prompt, gen=gen, e2e=e2e))
        d = jm.delta(prev, cur)
        prev = cur
        stats.add_minute(d, jm.is_busy(d, cur))
        old_rule_busy += int(jm.in_flight(cur) > 0)
    assert old_rule_busy == 0  # what the old watchdog saw
    assert stats.busy_minutes == 7
    assert stats.totals["vllm:request_success_total"] == 7
    assert stats.totals["vllm:generation_tokens_total"] == 2322

    t0 = 1_000_000.0
    line = stats.summary(job_start=t0, ready_at=t0 + 54 * 60, end=t0 + 69 * 60)
    assert line.startswith(
        "[Summary] allocation=69.0min startup=54.0min serving=15.0min"
    )
    assert "busy_minutes=7/16 (44%)" in line
    assert "requests=7" in line
    assert "request_s=33" in line
    assert "of serving" in line and "of allocation" in line


def test_summary_when_the_endpoint_never_came_up():
    line = jm.JobStats().summary(job_start=0.0, ready_at=None, end=600.0)
    assert line == "[Summary] allocation=10.0min; the endpoint never became ready"
