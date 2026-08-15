#!/usr/bin/env bash
# Verification of every stored brute-force result against the CPU heuristic. No GPU required.
#
# Recomputes each returned energy from scratch in float64, runs the NumPy simulated
# bifurcation solver on the same instance, and writes the bf_sbm_verification_* tables. With
# --check it also acts as a regression gate, exiting non-zero if the heuristic failed to reach
# the certified optimum on any instance.
#
# A couple of minutes for the full set of stored results.
#
# Usage:
#   ./scripts/run_verification.sh
#   ./scripts/run_verification.sh --mode distributed
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)    EXTRA+=(--mode "$2"); shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

LOG="${LOG_DIR}/verification_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

bench_python verify_sbm.py --check "${EXTRA[@]}"
log "done; tables in ${BENCHMARKS_DIR}/"
