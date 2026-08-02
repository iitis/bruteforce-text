#!/usr/bin/env bash
# Check that both nodes can actually run the experiments, before committing days of GPU time.
#
# Verifies, on the head and on the worker: the conda environment activates, the plugin and its
# dependencies import, the expected number of GPUs is visible, and the CUDA toolkit and driver
# versions match. A mismatch between the two nodes is worth knowing about in advance, because
# Ray will happily schedule subproblems onto a worker whose build differs from the head's.
#
# Usage: ./scripts/preflight.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

failures=0
check() {
    local label="$1"; shift
    if "$@" >/tmp/preflight.$$ 2>&1; then
        printf '  %-42s OK\n' "${label}"
    else
        printf '  %-42s FAILED\n' "${label}"
        sed 's/^/      /' /tmp/preflight.$$ | tail -5
        failures=$((failures + 1))
    fi
    rm -f /tmp/preflight.$$
}

log "head node: ${HEAD_IP}"
check "conda environment activates" bash -lc "${ACTIVATE_CMD}"
check "omnisolver-bruteforce imports" bash -lc \
    "${ACTIVATE_CMD} && ${PYTHON} -c 'import omnisolver.bruteforce.gpu.sampler'"
check "distributed sampler imports (needs ray)" bash -lc \
    "${ACTIVATE_CMD} && ${PYTHON} -c 'import omnisolver.bruteforce.gpu.distributed'"
check "dimod, numpy, numba import" bash -lc \
    "${ACTIVATE_CMD} && ${PYTHON} -c 'import dimod, numpy, numba'"
check "CUDA device visible" bash -lc \
    "${ACTIVATE_CMD} && ${PYTHON} -c 'from numba import cuda; assert cuda.is_available()'"
check "nvidia-smi reports ${GPUS_PER_NODE} GPU(s)" bash -lc \
    "test \$(nvidia-smi -L | wc -l) -eq ${GPUS_PER_NODE}"

log "worker node: ${WORKER_IP}"
check "ssh reachable (no password)" ${SSH} "${WORKER_IP}" true
if ${SSH} "${WORKER_IP}" true >/dev/null 2>&1; then
    check "conda environment activates" worker_run "${ACTIVATE_CMD}"
    check "omnisolver-bruteforce imports" worker_run \
        "${ACTIVATE_CMD} && ${PYTHON} -c 'import omnisolver.bruteforce.gpu.sampler'"
    check "distributed sampler imports (needs ray)" worker_run \
        "${ACTIVATE_CMD} && ${PYTHON} -c 'import omnisolver.bruteforce.gpu.distributed'"
    check "CUDA device visible" worker_run \
        "${ACTIVATE_CMD} && ${PYTHON} -c 'from numba import cuda; assert cuda.is_available()'"
    check "nvidia-smi reports ${GPUS_PER_NODE} GPU(s)" worker_run \
        "test \$(nvidia-smi -L | wc -l) -eq ${GPUS_PER_NODE}"
else
    log "skipping worker checks; only single-node topologies (1xM) will work"
    failures=$((failures + 1))
fi

log "toolchain comparison"
head_env="$(bash -lc "${ACTIVATE_CMD} && cd '${BENCHMARKS_DIR}' && ${PYTHON} bench_common.py environment" 2>/dev/null)"
worker_env="$(worker_run "${ACTIVATE_CMD} && cd '${WORKER_REPO_DIR}/benchmarks' && ${PYTHON} bench_common.py environment" 2>/dev/null)"
summarize() {
    "${PYTHON}" - "$1" <<'PY' 2>/dev/null || echo "      (could not parse)"
import json, sys
try:
    env = json.loads(sys.argv[1])
except Exception:
    print("      (no data)"); raise SystemExit(0)
cuda = env.get("cuda", {})
devices = ", ".join(sorted({d["name"] for d in cuda.get("devices", [])})) or "none"
print(f"      python {env.get('python')}  cuda runtime {cuda.get('runtime')}  "
      f"driver {cuda.get('driver')}")
print(f"      gpus: {devices}")
pkgs = env.get("packages", {})
print("      " + "  ".join(f"{k}={v}" for k, v in pkgs.items() if v))
PY
}
echo "  head:"   ; summarize "${head_env}"
echo "  worker:" ; summarize "${worker_env}"

if [ "${failures}" -eq 0 ]; then
    log "preflight passed"
else
    log "preflight found ${failures} problem(s); fix them before launching long runs"
fi
exit "${failures}"
