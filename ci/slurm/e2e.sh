#!/bin/bash
# End-to-end test of the real submission path on a throwaway Slurm cluster (slurm-docker-cluster):
#
#   bin/start.py --time → sbatch → bin/vllm.sh → src/orchestrator.py → vLLM (smollm-cpu's image and
#   args, SmolLM-135M on CPU) → LiteLLM proxy → run/endpoint.env → a chat completion through the
#   proxy → idle watchdog → [Summary]
#
# Nothing is faked or substituted: the compute nodes run the latest Apptainer release (which also
# provides `singularity`), installed by the workflow, so the job pulls and runs smollm-cpu's
# docker:// image exactly as on the real cluster.
#
# Runs inside the slurmctld container, from /data/repo (the shared job directory, so the compute
# nodes see the same files). Expects uv at /data/bin/uv. See .github/workflows/tests.yml.
set -euo pipefail

REPO=/data/repo
WORK=/data/ci
mkdir -p "$WORK"
cd "$REPO"

step() { echo; echo "=== $* ($(date +%T))"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

step "Test profile"
cat ci/slurm/models-ci.yaml >> config/models.yaml

step "Python environment (shared with the compute nodes through /data)"
/data/bin/uv venv -q --python /usr/bin/python3 .venv
/data/bin/uv pip install -q --python .venv/bin/python pyyaml requests 'litellm[proxy]==1.86.2'
# shellcheck disable=SC1091
source .venv/bin/activate

step "Submit through start.py with --time"
python -u bin/start.py smollm-cpu-ci --time 00:15:00 > "$WORK/start.log" 2>&1 &
START_PID=$!
for _ in $(seq 1 60); do
    JOB=$(sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p' "$WORK/start.log" | head -1)
    [ -n "$JOB" ] && break
    sleep 2
done
[ -n "${JOB:-}" ] || { cat "$WORK/start.log"; fail "no job submitted"; }
echo "job $JOB"
LOG="$REPO/vllm_${JOB}.log"

step "The time limit came from --time, not the profile"
LIMIT=$(scontrol show job "$JOB" | sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p')
echo "TimeLimit=$LIMIT"
[ "$LIMIT" = "00:15:00" ] || fail "expected TimeLimit=00:15:00, got $LIMIT"

step "Wait for the endpoint (Apptainer converts the image, vLLM downloads the model and starts)"
for _ in $(seq 1 240); do
    [ -f run/endpoint.env ] && break
    squeue -h -j "$JOB" | grep -q . || { cat "$LOG" 2>/dev/null; fail "job $JOB ended before publishing"; }
    sleep 5
done
[ -f run/endpoint.env ] || { cat "$LOG"; fail "endpoint.env never appeared"; }
cat run/endpoint.env
# shellcheck disable=SC1091
source run/endpoint.env

step "A chat completion through the LiteLLM proxy"
for _ in $(seq 1 30); do  # LiteLLM can lag a few seconds behind the endpoint file
    # CPU inference is slow, and the first request also warms the model up: allow minutes, not seconds.
    if REPLY=$(curl -sf --max-time 300 "$OPENAI_BASE_URL/chat/completions" \
        -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
        -d '{"model":"my-local-model","messages":[{"role":"user","content":"Say hello."}],"max_tokens":8}'); then
        break
    fi
    sleep 2
done
echo "${REPLY:-}"
echo "${REPLY:-}" | python -c 'import json,sys; r = json.load(sys.stdin); assert r["choices"][0]["message"]["content"].strip(), r; assert r["usage"]["completion_tokens"] > 0, r' \
    || fail "no completion through the proxy"

step "Wait for the idle watchdog to end the job"
for _ in $(seq 1 120); do
    squeue -h -j "$JOB" | grep -q . || break
    sleep 5
done
squeue -h -j "$JOB" | grep -q . && { scancel "$JOB"; fail "job $JOB still running after 10 minutes"; }
wait "$START_PID" || true
sleep 2  # let the job's last lines reach the shared log

step "Job log"
cat "$LOG"

step "Assertions"
grep -q "\[Orchestrator\] Endpoint published" "$LOG" || fail "endpoint not published"
# At least one: LiteLLM retries an upstream call that outlasts its timeout, so a slow first CPU
# completion can reach vLLM twice for the test's single request (seen in CI: reqs=+2).
grep -qE "\[Metrics\] reqs=\+[1-9][0-9]* " "$LOG" || fail "the request was not counted in a [Metrics] minute"
grep -q "\[Watchdog\] Max idle time (2 mins) reached" "$LOG" || fail "the watchdog did not end the job on idle"
SUMMARY=$(grep "\[Summary\]" "$LOG" || true)
[ -n "$SUMMARY" ] || fail "no [Summary] line"
echo "$SUMMARY" | grep -qE "requests=[1-9][0-9]* " || fail "summary counts no requests: $SUMMARY"
# One or more: a retried request can land in the minute after the original.
echo "$SUMMARY" | grep -qE "busy_minutes=[1-9][0-9]*/" || fail "summary shows no busy minute: $SUMMARY"
grep -q "\[Orchestrator\] Shutting down" "$LOG" || fail "no clean shutdown"

echo
echo "PASS: job $JOB — submit, --time, endpoint, proxy round trip, per-minute metrics, idle shutdown, summary"
