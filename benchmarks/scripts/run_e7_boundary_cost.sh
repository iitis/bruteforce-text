#!/usr/bin/env bash
# E7 - measured cost of returning partial results across one node boundary.
#
# The primary measurement times payload-bearing and matched discard tasks end to end on the
# local node, remote node and an even mixture, following Ray's ordinary task-return path.
# Defaults sweep 1, 100 and 1000 states and retain five raw repetitions. Payload-byte caps
# keep the default run practical; effective counts are recorded in the result.
#
# Needs a two-node cluster and a few minutes. 2x1 is enough; the GPUs are not used.
#
# Usage:
#   ./scripts/run_e7_boundary_cost.sh
#   ./scripts/run_e7_boundary_cost.sh --topology 2x4
#   ./scripts/run_e7_boundary_cost.sh --smoke
#   ./scripts/run_e7_boundary_cost.sh --batch-sizes "16 64"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

TOPOLOGY=2x1
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --topology)        TOPOLOGY="$2"; shift 2 ;;
        --batch-sizes)
            read -r -a E7_VALUES <<< "${2//,/ }"
            EXTRA+=(--batch-sizes "${E7_VALUES[@]}")
            shift 2
            ;;
        --num-states)
            read -r -a E7_VALUES <<< "${2//,/ }"
            EXTRA+=(--num-states "${E7_VALUES[@]}")
            shift 2
            ;;
        --num-tasks)       EXTRA+=(--num-tasks "$2"); shift 2 ;;
        --max-batch-bytes) EXTRA+=(--max-batch-bytes "$2"); shift 2 ;;
        --repeats)         EXTRA+=(--repeats "$2"); shift 2 ;;
        --operation-timeout-seconds)
            EXTRA+=(--operation-timeout-seconds "$2"); shift 2 ;;
        --run-timeout-seconds) EXTRA+=(--run-timeout-seconds "$2"); shift 2 ;;
        --smoke)           EXTRA+=(--smoke); shift ;;
        --run-id)          EXTRA+=(--run-id "$2"); shift 2 ;;
        -h|--help)         sed -n '2,16p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e7_boundary_cost_${TOPOLOGY}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

cleanup() {
    "${SCRIPTS_DIR}/ray_cluster.sh" stop || true
}
trap cleanup EXIT

"${SCRIPTS_DIR}/ray_cluster.sh" start "${TOPOLOGY}"
export PYTHONUNBUFFERED=1
bench_python exp_boundary_cost.py --topology "${TOPOLOGY}" "${EXTRA[@]}"
cleanup
trap - EXIT

log "done; results in ${BENCHMARKS_DIR}/results/boundary_cost/"
