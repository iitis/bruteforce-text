#!/usr/bin/env bash
# E2 - weak scaling: the work per GPU is held constant while the GPU count grows.
#
# The size follows the GPU count so that every device always enumerates 2^base_size
# configurations: N = base + k with 2^k GPUs. Ideal weak scaling keeps the wall-clock time
# flat across the series; whatever growth remains is dispatch, merge and, for the 2xM rows,
# the node boundary.
#
# With base size 50 each point is roughly 40 minutes on H100-class GPUs.
#
# Usage:
#   ./scripts/run_e2_weak_scaling.sh
#   ./scripts/run_e2_weak_scaling.sh --base-size 48
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

BASE="${WEAK_SCALING_BASE_SIZE}"
TOPOS="${TOPOLOGIES}"
WITH_REFERENCE=1

while [ $# -gt 0 ]; do
    case "$1" in
        --base-size)    BASE="$2"; shift 2 ;;
        --topologies)   TOPOS="$2"; shift 2 ;;
        --no-reference) WITH_REFERENCE=0; shift ;;
        -h|--help)      sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e2_weak_scaling_base${BASE}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

if [ "${WITH_REFERENCE}" -eq 1 ]; then
    if result_exists "weak_scaling/base${BASE}_single-gpu.json"; then
        log "skipping single-GPU reference (result already present)"
    else
        log "single-GPU reference: N=${BASE}, k=0 (no Ray)"
        "${SCRIPTS_DIR}/ray_cluster.sh" stop
        CUDA_VISIBLE_DEVICES=0 bench_python exp_weak_scaling.py --base-size "${BASE}" --single-gpu
    fi
fi

for topology in ${TOPOS}; do
    if result_exists "weak_scaling/base${BASE}_${topology}.json"; then
        log "skipping ${topology} (result already present)"
        continue
    fi
    log "=== topology ${topology}, work per GPU 2^${BASE} ==="
    "${SCRIPTS_DIR}/ray_cluster.sh" start "${topology}"
    bench_python exp_weak_scaling.py --base-size "${BASE}" --topology "${topology}"
done

"${SCRIPTS_DIR}/ray_cluster.sh" stop

log "summary"
bench_python - <<'PY'
import bench_common as common

rows = []
for result in common.load_results("weak_scaling"):
    topology = result["topology"]
    rows.append(
        (
            topology.get("total_gpus", 1),
            topology["label"],
            result["num_variables"],
            topology.get("num_fixed_vars", 0),
            result["timings"].get("solve_time_in_seconds"),
        )
    )
if not rows:
    raise SystemExit("no results yet")

baseline = min(rows, key=lambda r: r[0])[4]
print(f"{'topology':>12} {'GPUs':>5} {'N':>4} {'k':>3} {'solve [s]':>12} "
      f"{'vs 1 GPU':>9}")
for gpus, label, size, k, solve in sorted(rows, key=lambda r: (r[0], r[1])):
    ratio = f"{solve / baseline:.2f}x" if baseline and solve else ""
    print(f"{label:>12} {gpus:>5} {size:>4} {k:>3} {solve:>12.2f} {ratio:>9}")
print("\nIdeal weak scaling would keep the last column at 1.00x.")
PY
