#!/usr/bin/env bash
# E8 - rerun dSB and simulated annealing against the certified family sweep.
#
# This is CPU-only and reads results/instance_families/N44.json by default. The Python driver
# refuses to replace an existing result unless --overwrite is passed explicitly.
#
# Usage:
#   ./scripts/run_e8_multi_solver.sh
#   ./scripts/run_e8_multi_solver.sh --seed-bases 420000 430000
#   ./scripts/run_e8_multi_solver.sh --num-reads 32 --num-sweeps 100 \
#       --output /tmp/multi_solver_smoke.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

LOG="${LOG_DIR}/e8_multi_solver_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

HAS_OUTPUT=0
for ARG in "$@"; do
    [ "${ARG}" = "--output" ] && HAS_OUTPUT=1
done
EXTRA=()
if [ "${HAS_OUTPUT}" -eq 0 ]; then
    EXTRA+=(--output "${BENCHMARKS_DIR}/results/multi_solver/run_$(date '+%Y%m%d_%H%M%S').json")
fi

bench_python exp_multi_solver.py "${EXTRA[@]}" "$@"

log "done; results in ${BENCHMARKS_DIR}/results/multi_solver/"
