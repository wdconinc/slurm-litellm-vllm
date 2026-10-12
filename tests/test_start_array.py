"""Offline tests for `start.sh --array`: no Slurm, no network (preflight and sbatch are stubbed)."""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bin"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import start

from src.api import SlurmLLMFleet


@pytest.mark.parametrize(
    "spec, want",
    [
        ("5", "1-5%1"),
        ("1", "1-1%1"),
        ("1-5", "1-5%1"),
        ("0-4", "0-4%1"),
        ("1-5%1", "1-5%1"),
        ("1-5%2", "1-5%1"),
        ("1,2,3", "1,2,3%1"),
        ("1-10:2", "1-10:2%1"),
        ("", ""),
    ],
)
def test_normalize_array_spec(spec, want):
    assert start.normalize_array_spec(spec) == want


@pytest.mark.parametrize(
    "argv, want",
    [
        (["qwen"], (None, ["qwen"])),
        ([], (None, [])),
        (["qwen", "--array", "1-5"], ("1-5%1", ["qwen"])),
        (["--array", "5", "qwen"], ("1-5%1", ["qwen"])),
        (["qwen", "--array=1-3%1"], ("1-3%1", ["qwen"])),
        (["qwen", "--array=0-4"], ("0-4%1", ["qwen"])),
    ],
)
def test_pop_array_arg(argv, want):
    assert start.pop_array_arg(argv) == want


def test_pop_array_and_time_combined():
    argv = ["qwen", "--time", "08:00:00", "--array", "1-5"]
    time_limit, rest = start.pop_time_arg(argv)
    array_spec, rest = start.pop_array_arg(rest)
    assert time_limit == "08:00:00"
    assert array_spec == "1-5%1"
    assert rest == ["qwen"]


def test_array_without_a_value_exits():
    with pytest.raises(SystemExit):
        start.pop_array_arg(["qwen", "--array"])


def _sbatch_cmd(time_limit=None, array_spec=None):
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
        job_id = start.submit_job("qwen", time_limit=time_limit, array_spec=array_spec)
    assert job_id == "12345"
    (cmd,) = [c for c in calls if c and c[0] == "sbatch"]
    return cmd


def test_array_added_to_sbatch():
    cmd = _sbatch_cmd(array_spec="1-5")
    assert "--array=1-5%1" in cmd
    assert sum(a.startswith("--array=") for a in cmd) == 1
    assert cmd[-2:] == ["bin/vllm.sh", "qwen"]


def test_array_and_time_added_to_sbatch():
    cmd = _sbatch_cmd(time_limit="06:00:00", array_spec="3")
    assert "--time=06:00:00" in cmd
    assert "--array=1-3%1" in cmd
    assert cmd[-2:] == ["bin/vllm.sh", "qwen"]


def test_slurm_llm_fleet_array():
    calls = []

    def fake_submit(model_key, time_limit=None, array_spec=None):
        calls.append((model_key, time_limit, array_spec))
        return "99999"

    with patch("src.api.submit_job", fake_submit):
        fleet = SlurmLLMFleet(model="smollm-cpu", time_limit="01:00:00", array="1-4")
        fleet.start(wait=False)
        assert fleet.job_id == "99999"
        assert calls == [("smollm-cpu", "01:00:00", "1-4")]
