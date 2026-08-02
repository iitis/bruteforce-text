#!/usr/bin/env bash
# E5 - QUBO instances alongside the Ising ones.
#
# Loads the same coefficient files once as SPIN and once as BINARY. The energies differ, since
# the same coefficients define different objectives; the runtimes should not, because both are
# solved by the identical QUBO code path. Runs on a single GPU by default.
#
# Usage:
#   ./scripts/run_e5_qubo.sh
#   ./scripts/run_e5_qubo.sh --distributed --topology 2x4
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SIZES="${QUBO_SIZES}"
DISTRIBUTED=0
TOPOLOGY=2x4

while [ $# -gt 0 ]; do
    case "$1" in
        --sizes)       SIZES="$2"; shift 2 ;;
        --distributed) DISTRIBUTED=1; shift ;;
        --topology)    TOPOLOGY="$2"; shift 2 ;;
        -h|--help)     sed -n '2,12p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/e5_qubo_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

if [ "${DISTRIBUTED}" -eq 1 ]; then
    "${SCRIPTS_DIR}/ray_cluster.sh" start "${TOPOLOGY}"
    # shellcheck disable=SC2086
    bench_python exp_qubo.py --sizes ${SIZES} --distributed --topology "${TOPOLOGY}"
    "${SCRIPTS_DIR}/ray_cluster.sh" stop
else
    "${SCRIPTS_DIR}/ray_cluster.sh" stop
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0 bench_python exp_qubo.py --sizes ${SIZES}
fi

log "done; results in ${BENCHMARKS_DIR}/results/qubo/"
