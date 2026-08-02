#!/usr/bin/env bash
# E1 - strong scaling: one problem size, every GPU count and node layout.
#
# Restarts the Ray cluster once per topology and runs one measurement on each, plus a
# single-GPU reference that does not involve Ray at all. The pairs 1x2/2x1 and 1x4/2x2 hold
# the GPU count fixed and differ only in whether a node boundary is crossed.
#
# At N=50 the whole series takes roughly 2 hours on H100-class GPUs: the 1-GPU points dominate
# it (about 35 minutes each) while 2x4 finishes in under 5 minutes.
#
# Usage:
#   ./scripts/run_e1_strong_scaling.sh
#   ./scripts/run_e1_strong_scaling.sh --size 48
#   ./scripts/run_e1_strong_scaling.sh --topologies "1x4 2x2"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SIZE="${STRONG_SCALING_SIZE}"
TOPOS="${TOPOLOGIES}"
WITH_REFERENCE=1

while [ $# -gt 0 ]; do
    case "$1" in
        --size)        SIZE="$2"; shift 2 ;;
        --topologies)  TOPOS="$2"; shift 2 ;;
        --no-reference) WITH_REFERENCE=0; shift ;;
        -h|--help)     sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e1_strong_scaling_N${SIZE}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

if [ "${WITH_REFERENCE}" -eq 1 ]; then
    if result_exists "strong_scaling/N${SIZE}_single-gpu.json"; then
        log "skipping single-GPU reference (result already present)"
    else
        log "single-GPU reference at N=${SIZE} (no Ray)"
        "${SCRIPTS_DIR}/ray_cluster.sh" stop
        CUDA_VISIBLE_DEVICES=0 bench_python exp_strong_scaling.py --size "${SIZE}" --single-gpu
    fi
fi

for topology in ${TOPOS}; do
    if result_exists "strong_scaling/N${SIZE}_${topology}.json"; then
        log "skipping ${topology} (result already present)"
        continue
    fi
    log "=== topology ${topology}, N=${SIZE} ==="
    "${SCRIPTS_DIR}/ray_cluster.sh" start "${topology}"
    bench_python exp_strong_scaling.py --size "${SIZE}" --topology "${topology}"
done

"${SCRIPTS_DIR}/ray_cluster.sh" stop

log "summary"
bench_python - <<'PY'
import bench_common as common

rows = []
for result in common.load_results("strong_scaling"):
    rows.append(
        (
            result["topology"].get("total_gpus", 1),
            result["topology"]["label"],
            result["num_variables"],
            result["timings"].get("solve_time_in_seconds"),
            result["timings"].get("merge_time_in_seconds"),
        )
    )
if not rows:
    raise SystemExit("no results yet")

reference = next((r for r in rows if r[1] == "single-gpu"), None)
print(f"{'topology':>12} {'GPUs':>5} {'N':>4} {'solve [s]':>12} {'merge [s]':>10} "
      f"{'speedup':>8} {'efficiency':>11}")
for gpus, label, size, solve, merge in sorted(rows, key=lambda r: (r[2], r[0], r[1])):
    speedup = efficiency = ""
    if reference and reference[2] == size and solve:
        s = reference[3] / solve
        speedup, efficiency = f"{s:.2f}", f"{100 * s / gpus:.1f}%"
    merge_text = f"{merge:.4f}" if merge is not None else "n/a"
    print(f"{label:>12} {gpus:>5} {size:>4} {solve:>12.2f} {merge_text:>10} "
          f"{speedup:>8} {efficiency:>11}")
PY
