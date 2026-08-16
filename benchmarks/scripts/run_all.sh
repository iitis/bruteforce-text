#!/usr/bin/env bash
# Run every experiment needed to complete the revision, cheapest first.
#
# The order is deliberate: the checks and the CPU-only experiments come first, so that a broken
# environment or a broken assumption surfaces in minutes rather than after a day of GPU time.
# The multi-day Fig. 1 sweep is excluded by default; pass --with-figure to include it, or run
# scripts/run_manuscript_figure.sh separately under tmux.
#
# Total without --with-figure: roughly 9 to 10 hours, dominated by E2 (~4 h) and E1 (~2 h).
#
# Usage:
#   ./scripts/run_all.sh                  # preflight + verification + E8 + E3 + E7 + E6 + E4 + E5 + E1 + E2
#   ./scripts/run_all.sh --cpu-only       # only what needs no GPU
#   ./scripts/run_all.sh --with-figure    # ... and the multi-day Fig. 1 sweep at the end
#   ./scripts/run_all.sh --skip preflight,e2
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

WITH_FIGURE=0
CPU_ONLY=0
SKIP=""

while [ $# -gt 0 ]; do
    case "$1" in
        --with-figure) WITH_FIGURE=1; shift ;;
        --cpu-only)    CPU_ONLY=1; shift ;;
        --skip)        SKIP="$2"; shift 2 ;;
        -h|--help)     sed -n '2,16p' "$0"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

skipped() { case ",${SKIP}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

LOG="${LOG_DIR}/run_all_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
log "logging to ${LOG}"

declare -a COMPLETED=() FAILED=()
stage() {
    local name="$1"; shift
    if skipped "${name}"; then
        log "--- skipping ${name} (requested) ---"
        return 0
    fi
    log "=== ${name} ==="
    if "$@"; then
        COMPLETED+=("${name}")
    else
        FAILED+=("${name}")
        log "!!! ${name} failed; continuing with the remaining stages"
    fi
}

# Cheap and diagnostic first.
stage preflight    "${SCRIPTS_DIR}/preflight.sh"
stage verification "${SCRIPTS_DIR}/run_verification.sh"
stage e8           "${SCRIPTS_DIR}/run_e8_multi_solver.sh"
stage e3           "${SCRIPTS_DIR}/run_e3_controller_cost.sh"
# E7 needs two nodes, so it sits with the GPU stages even though it uses no GPU.

if [ "${CPU_ONLY}" -eq 0 ]; then
    # Short GPU experiments next: each answers one reviewer point in well under an hour.
    stage e7 "${SCRIPTS_DIR}/run_e7_boundary_cost.sh"       # ~5 min, 2 nodes
    stage e6 "${SCRIPTS_DIR}/run_e6_instance_families.sh"   # ~18 min
    stage e4 "${SCRIPTS_DIR}/run_e4_precision.sh"           # ~30 min
    stage e5 "${SCRIPTS_DIR}/run_e5_qubo.sh"                # ~1.6 h
    # The scaling series restart the cluster once per topology and are the expensive ones.
    stage e1 "${SCRIPTS_DIR}/run_e1_strong_scaling.sh"      # ~2 h
    stage e2 "${SCRIPTS_DIR}/run_e2_weak_scaling.sh"        # ~4 h
    if [ "${WITH_FIGURE}" -eq 1 ]; then
        stage figure "${SCRIPTS_DIR}/run_manuscript_figure.sh"
    else
        log "--- skipping the Fig. 1 sweep (multi-day); pass --with-figure to include it ---"
    fi
else
    log "--- skipping all GPU experiments (--cpu-only) ---"
fi

"${SCRIPTS_DIR}/ray_cluster.sh" stop || true

log "completed: ${COMPLETED[*]:-none}"
if [ "${#FAILED[@]}" -gt 0 ]; then
    log "failed: ${FAILED[*]}"
    exit 1
fi
log "all requested stages completed"
