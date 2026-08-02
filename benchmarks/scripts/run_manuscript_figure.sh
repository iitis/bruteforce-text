#!/usr/bin/env bash
# The original benchmark sweep behind Fig. 1 and Table 3 of the manuscript.
#
# Two sweeps on the same instances: the distributed sampler on all eight GPUs (2x4) and the
# single-GPU sampler on one of them. Existing result files are skipped, so the published data
# is never silently overwritten and an interrupted sweep can simply be restarted.
#
# Cost: with the published results present every size is skipped and this only redraws the
# figure, in seconds. It is the long one only for points that are missing, because each added
# variable doubles the runtime: on 8x H100 the distributed sweep reaches ~4.7 h at N=56, ~19 h
# at N=58 and ~3.15 days at N=60, and the single-GPU sweep ~9.4 h at N=54. If you are
# regenerating data, run it under tmux/screen and use --max-size to stop earlier.
#
# Usage:
#   ./scripts/run_manuscript_figure.sh                      # full sweep, both modes
#   ./scripts/run_manuscript_figure.sh --max-size 50        # stop at N=50
#   ./scripts/run_manuscript_figure.sh --mode single-gpu    # one mode only
#   ./scripts/run_manuscript_figure.sh --plot-only          # just redraw the figure
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

MIN_SIZE=40
MAX_SIZE=60
SINGLE_GPU_MAX_SIZE=54
MODES="distributed single-gpu"
PLOT_ONLY=0
SEED=42

while [ $# -gt 0 ]; do
    case "$1" in
        --max-size) MAX_SIZE="$2"; shift 2 ;;
        --min-size) MIN_SIZE="$2"; shift 2 ;;
        --mode)     MODES="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --plot-only) PLOT_ONLY=1; shift ;;
        -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/manuscript_figure_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

if [ "${PLOT_ONLY}" -eq 0 ]; then
    for mode in ${MODES}; do
        stop=${MAX_SIZE}
        if [ "${mode}" = "single-gpu" ] && [ "${stop}" -gt "${SINGLE_GPU_MAX_SIZE}" ]; then
            stop=${SINGLE_GPU_MAX_SIZE}
            log "capping the single-GPU sweep at N=${stop} (beyond that it takes weeks)"
        fi

        if [ "${mode}" = "distributed" ]; then
            "${SCRIPTS_DIR}/ray_cluster.sh" start 2x4
        else
            # The single-GPU sampler does not use Ray; stop it so the two modes do not
            # compete for the same device.
            "${SCRIPTS_DIR}/ray_cluster.sh" stop
        fi

        log "sweeping N=${MIN_SIZE}..${stop} step 2 with the ${mode} sampler"
        if [ "${mode}" = "single-gpu" ]; then
            CUDA_VISIBLE_DEVICES=0 bench_python bf.py \
                --start "${MIN_SIZE}" --stop "${stop}" --step 2 \
                --sampler-mode "${mode}" --seed "${SEED}" --skip-existing
        else
            bench_python bf.py \
                --start "${MIN_SIZE}" --stop "${stop}" --step 2 \
                --sampler-mode "${mode}" --seed "${SEED}" --skip-existing
        fi
    done
    "${SCRIPTS_DIR}/ray_cluster.sh" stop
fi

log "rendering Fig. 1"
bench_python plot_distributed.py

log "done; results in ${BENCHMARKS_DIR}/results/{distributed,single-gpu}/"
