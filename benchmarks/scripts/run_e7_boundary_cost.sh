#!/usr/bin/env bash
# E7 - per-unit cost of one node boundary, used to bound the controller at large P.
#
# The controller sweep of E3 runs on a single machine and therefore contains no network
# transfer: its numbers are a lower bound on the controller cost, not an estimate. This
# experiment measures the two unit costs that carry the network - dispatching a task to
# another node, and collecting a partial result from it - so that the merge at large P can be
# bounded by unit-cost x P, with the assumption of no saturation stated explicitly.
#
# Needs a two-node cluster and a couple of minutes. 2x1 is enough; the GPUs are not used.
#
# Usage:
#   ./scripts/run_e7_boundary_cost.sh
#   ./scripts/run_e7_boundary_cost.sh --topology 2x4
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

TOPOLOGY=2x1
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --topology)     TOPOLOGY="$2"; shift 2 ;;
        --num-objects)  EXTRA+=(--num-objects "$2"); shift 2 ;;
        --num-tasks)    EXTRA+=(--num-tasks "$2"); shift 2 ;;
        -h|--help)      sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e7_boundary_cost_${TOPOLOGY}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

"${SCRIPTS_DIR}/ray_cluster.sh" start "${TOPOLOGY}"
bench_python exp_boundary_cost.py --topology "${TOPOLOGY}" "${EXTRA[@]}"
"${SCRIPTS_DIR}/ray_cluster.sh" stop

log "done; results in ${BENCHMARKS_DIR}/results/boundary_cost/"
