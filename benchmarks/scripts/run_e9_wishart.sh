#!/usr/bin/env bash
# E9 - Wishart planted family: the heuristic fails at small alpha, certification decides.
#
# For every (alpha, replica): derive the deterministic planted instance, brute-force it on one
# GPU, check the certificate against the closed-form ground-state energy E_0, run the shipped
# CPU dSB heuristic under the E6/E8 budgets, and record whether it reached the certified
# optimum. Small alpha is the hard regime; alpha = 0.5 is the easy control. The alpha sweep
# also feeds the success-fraction figure (plot_wishart.py).
#
# Roughly 40 minutes at N=40 for 140 instances: ~2 s of GPU per certificate, ~15 s of CPU per
# dSB run. The analytic E_0 also allows a GPU-free re-check via --skip-bruteforce.
#
# Usage:
#   ./scripts/run_e9_wishart.sh
#   ./scripts/run_e9_wishart.sh --size 40 --replicas 20 --alphas "0.2 0.3 0.5"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SIZE="${WISHART_SIZE}"
REPLICAS="${WISHART_REPLICAS}"
ALPHAS="${WISHART_ALPHAS}"

while [ $# -gt 0 ]; do
    case "$1" in
        --size)     SIZE="$2"; shift 2 ;;
        --replicas) REPLICAS="$2"; shift 2 ;;
        --alphas)   ALPHAS="$2"; shift 2 ;;
        -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e9_wishart_N${SIZE}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

"${SCRIPTS_DIR}/ray_cluster.sh" stop
log "N=${SIZE}, ${REPLICAS} instance(s) per alpha: ${ALPHAS}"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 bench_python exp_wishart.py \
    --size "${SIZE}" --replicas "${REPLICAS}" --alphas ${ALPHAS}

log "done; results in ${BENCHMARKS_DIR}/results/wishart/"
