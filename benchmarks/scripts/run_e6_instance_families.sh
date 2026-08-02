#!/usr/bin/env bash
# E6 - certified verification of the heuristic across instance families.
#
# For every instance: brute-force it on one GPU, recompute the energy in float64, run the CPU
# simulated bifurcation solver, and record whether the heuristic recovered the certified
# optimum. Sweeps uniform, bimodal, Gaussian and sparse couplings.
#
# Roughly 11 minutes at N=44 for 20 instances on an H100; the brute-force cost doubles per
# added variable, so N=48 is about 3 hours.
#
# Usage:
#   ./scripts/run_e6_instance_families.sh
#   ./scripts/run_e6_instance_families.sh --size 48 --replicas 5
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SIZE="${FAMILIES_SIZE}"
REPLICAS="${FAMILIES_REPLICAS}"
FAMILIES="uniform bimodal gaussian sparse"

while [ $# -gt 0 ]; do
    case "$1" in
        --size)     SIZE="$2"; shift 2 ;;
        --replicas) REPLICAS="$2"; shift 2 ;;
        --families) FAMILIES="$2"; shift 2 ;;
        -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e6_instance_families_N${SIZE}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

"${SCRIPTS_DIR}/ray_cluster.sh" stop
log "N=${SIZE}, ${REPLICAS} instance(s) per family: ${FAMILIES}"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 bench_python exp_instance_families.py \
    --size "${SIZE}" --replicas "${REPLICAS}" --families ${FAMILIES}

log "done; results in ${BENCHMARKS_DIR}/results/instance_families/"
