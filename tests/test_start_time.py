"""Offline tests for `start.sh --time`: no Slurm, no network (preflight and sbatch are stubbed)."""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bin"))

import start


@pytest.mark.parametrize(
    "argv, want",
    [
        (["qwen"], (None, ["qwen"])),
        ([], (None, [])),
        (["qwen", "--time", "08:00:00"], ("08:00:00", ["qwen"])),
        (["--time", "1-00:00:00", "qwen"], ("1-00:00:00", ["qwen"])),
        (["qwen", "--time=12:00:00"], ("12:00:00", ["qwen"])),
    ],
)
def test_pop_time_arg(argv, want):
    assert start.pop_time_arg(argv) == want


def test_time_without_a_value_exits():
    with pytest.raises(SystemExit):
        start.pop_time_arg(["qwen", "--time"])


def _sbatch_cmd(time_limit):
    """The sbatch argv submit_job builds for the qwen profile."""
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Submitted batch job 12345\n", stderr=""
        )

    with (
        patch.object(start, "run_preflight_checks"),
        patch.object(start.subprocess, "run", fake_run),
    ):
        job_id = start.submit_job("qwen", time_limit)
    assert job_id == "12345"
    (cmd,) = [c for c in calls if c and c[0] == "sbatch"]
    return cmd


def test_time_overrides_the_profile():
    cmd = _sbatch_cmd("10:00:00")
    assert "--time=10:00:00" in cmd
    assert sum(a.startswith("--time=") for a in cmd) == 1
    assert cmd[-2:] == ["bin/vllm.sh", "qwen"]


def test_without_time_the_profile_value_is_used():
    cmd = _sbatch_cmd(None)
    assert "--time=04:00:00" in cmd
