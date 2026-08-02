#!/usr/bin/env bash
# E4 - single precision against double precision on the ground-state path.
#
# Single GPU, no Ray. Note that float64 is a different numerical path, not merely a slower
# one: the compensated updates and the periodic re-anchoring are enabled only for float32 and
# only from 40 variables per kernel upwards.
#
# Budget: the float32 runs follow the doubling rule (~3 s at N=40 on an H100), the float64 runs
# are several times slower on data-centre GPUs and far worse on consumer cards.
#
# Usage:
#   ./scripts/run_e4_precision.sh
#   ./scripts/run_e4_precision.sh --sizes "40 42"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SIZES="${PRECISION_SIZES}"
while [ $# -gt 0 ]; do
    case "$1" in
        --sizes)   SIZES="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e4_precision_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

"${SCRIPTS_DIR}/ray_cluster.sh" stop
log "comparing float32 and float64 at N=${SIZES}"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 bench_python exp_precision.py --sizes ${SIZES}

log "done; results in ${BENCHMARKS_DIR}/results/precision/"
