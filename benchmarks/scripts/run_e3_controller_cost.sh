#!/usr/bin/env bash
# E3 - controller cost as a function of the number of subproblems. No GPU required.
#
# The number of subproblems is set by num_fixed_vars, not by the device count, so the
# controller can be driven on a CPU with the partial results workers would have returned.
# That is what makes the "would the merge limit us at 2^16 workers?" question answerable on
# this hardware.
#
# Minutes for --max-k 14; 2^16 subproblems needs a few GB of RAM.
#
# Usage:
#   ./scripts/run_e3_controller_cost.sh
#   ./scripts/run_e3_controller_cost.sh --max-k 14
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

MAX_K="${CONTROLLER_MAX_K}"
MIN_K=3
REPEATS=5

while [ $# -gt 0 ]; do
    case "$1" in
        --max-k)   MAX_K="$2"; shift 2 ;;
        --min-k)   MIN_K="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e3_controller_cost_k${MIN_K}-${MAX_K}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

# This experiment uses Ray only as a task scheduler on the local machine, so any cluster
# started for the GPU experiments is stopped first to keep the measurement clean.
"${SCRIPTS_DIR}/ray_cluster.sh" stop

log "measuring the controller for 2^${MIN_K}..2^${MAX_K} subproblems"
bench_python exp_controller_cost.py --min-k "${MIN_K}" --max-k "${MAX_K}" --repeats "${REPEATS}"

log "done; results in ${BENCHMARKS_DIR}/results/controller_cost/"
