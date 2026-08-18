#!/usr/bin/env bash
# E9 - Wishart planted family: the heuristic fails at small alpha, certification decides.
#
# For every (alpha, replica): derive the deterministic planted instance, brute-force it on one
# GPU, check the certificate against the closed-form ground-state energy E_0, run the shipped
# CPU dSB heuristic under the E6/E8 budgets, and record whether it reached the certified
# optimum. Small alpha is the hard regime; alpha = 0.5 is the easy control. Runs both the
# planted arm (closed-form E_0 checks every certificate) and the paired unplanted arm (the
# optimum is unknown a priori; certificates must clear the analytic lower bound). The sweep
# feeds the success-fraction figure (plot_wishart.py).
#
# Roughly 1 hour at N=40 for 2 x 140 instances: ~2 s of GPU per certificate, ~8-15 s of CPU
# per dSB run. The planted arm allows a GPU-free re-check via
#   exp_wishart.py --ensembles planted --skip-bruteforce
#
# Usage:
#   ./scripts/run_e9_wishart.sh
#   ./scripts/run_e9_wishart.sh --replicas 20 --alphas "0.2 0.3 0.5" --ensembles planted
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SIZE="${WISHART_SIZE}"
REPLICAS="${WISHART_REPLICAS}"
ALPHAS="${WISHART_ALPHAS}"
ENSEMBLES="${WISHART_ENSEMBLES}"

while [ $# -gt 0 ]; do
    case "$1" in
        --size)      SIZE="$2"; shift 2 ;;
        --replicas)  REPLICAS="$2"; shift 2 ;;
        --alphas)    ALPHAS="$2"; shift 2 ;;
        --ensembles) ENSEMBLES="$2"; shift 2 ;;
        -h|--help)   sed -n '2,18p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e9_wishart_N${SIZE}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

"${SCRIPTS_DIR}/ray_cluster.sh" stop
log "N=${SIZE}, ${REPLICAS} instance(s) per alpha: ${ALPHAS}; ensembles: ${ENSEMBLES}"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 bench_python exp_wishart.py \
    --size "${SIZE}" --replicas "${REPLICAS}" --alphas ${ALPHAS} --ensembles ${ENSEMBLES}

log "done; results in ${BENCHMARKS_DIR}/results/wishart/"
