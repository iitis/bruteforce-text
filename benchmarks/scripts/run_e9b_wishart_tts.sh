#!/usr/bin/env bash
# E9b - per-instance time-to-solution on the hard Wishart point (alpha = 0.2).
#
# Repeats the shipped dSB heuristic many times per instance (independent seeds, documented
# per-run budget) on both paired arms, estimating each instance's success probability p_i.
# TTS_99(i) = t_run * ceil(ln 0.01 / ln(1 - p_i)) then compares against the float64
# brute-force certification time, which is why this launcher certifies with --bf-dtype
# double: single-precision ranking alone does not prove that no better state was rejected.
#
# Cost: ~2 s of GPU per certificate, then repeats x instances dSB runs on CPU. At the
# defaults (100 repeats, 20 instances x 2 ensembles, ~10 s/run) this is an overnight
# ~11 h CPU job; --repeats 50 halves it. No GPU is held during the dSB phase.
#
# Usage:
#   ./scripts/run_e9b_wishart_tts.sh
#   ./scripts/run_e9b_wishart_tts.sh --repeats 50 --alphas "0.2 0.25"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SIZE="${WISHART_SIZE}"
REPLICAS="${WISHART_REPLICAS}"
ALPHAS="${WISHART_TTS_ALPHAS}"
REPEATS="${WISHART_TTS_REPEATS}"
ENSEMBLES="${WISHART_ENSEMBLES}"

while [ $# -gt 0 ]; do
    case "$1" in
        --size)      SIZE="$2"; shift 2 ;;
        --replicas)  REPLICAS="$2"; shift 2 ;;
        --alphas)    ALPHAS="$2"; shift 2 ;;
        --repeats)   REPEATS="$2"; shift 2 ;;
        --ensembles) ENSEMBLES="$2"; shift 2 ;;
        -h|--help)   sed -n '2,17p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e9b_wishart_tts_N${SIZE}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

"${SCRIPTS_DIR}/ray_cluster.sh" stop
log "N=${SIZE}, ${REPLICAS} instance(s) per alpha: ${ALPHAS}; ensembles: ${ENSEMBLES}; ${REPEATS} dSB repeats per instance"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 bench_python exp_wishart.py \
    --size "${SIZE}" --replicas "${REPLICAS}" --alphas ${ALPHAS} --ensembles ${ENSEMBLES} \
    --heuristic-repeats "${REPEATS}" --bf-dtype double

log "done; results in ${BENCHMARKS_DIR}/results/wishart/"
