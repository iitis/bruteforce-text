#!/usr/bin/env bash
# Start, stop and inspect the Ray cluster in a requested GPU topology.
#
# A topology is written <nodes>x<gpus per node>: 2x4 is the eight-GPU configuration behind
# Fig. 1 of the manuscript, 2x1 is one GPU on each of the two nodes. Fewer GPUs per node are
# obtained by masking with CUDA_VISIBLE_DEVICES before starting Ray, which is what makes
# 1x2 and 2x1 (or 1x4 and 2x2) directly comparable: same number of GPUs, different number of
# node boundaries.
#
# Usage:
#   ./scripts/ray_cluster.sh start 2x4
#   ./scripts/ray_cluster.sh status
#   ./scripts/ray_cluster.sh stop
#
# Run from the head node.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

usage() {
    cat >&2 <<EOF
usage: $(basename "$0") {start <NxM>|stop|status}

  start <NxM>  start a cluster with N nodes and M GPUs each (N is 1 or 2)
  stop         stop Ray on both nodes
  status       print the detected GPU topology as JSON
EOF
    exit 2
}

parse_topology() {
    local label="$1"
    [[ "${label}" =~ ^([0-9]+)x([0-9]+)$ ]] || die "malformed topology '${label}', expected NxM"
    NODES="${BASH_REMATCH[1]}"
    GPUS="${BASH_REMATCH[2]}"
    [ "${NODES}" -ge 1 ] && [ "${NODES}" -le 2 ] || die "only 1 or 2 nodes are configured, got ${NODES}"
    [ "${GPUS}" -ge 1 ] && [ "${GPUS}" -le "${GPUS_PER_NODE}" ] ||
        die "each node has ${GPUS_PER_NODE} GPU(s), cannot request ${GPUS}"
}

# Comma-separated device list 0,1,...,M-1
device_mask() {
    local count="$1" list=""
    for ((i = 0; i < count; i++)); do list+="${i},"; done
    printf '%s' "${list%,}"
}

cmd_stop() {
    log "stopping Ray on head (${HEAD_IP})"
    ray stop --force >/dev/null 2>&1 || true
    if ${SSH} "${WORKER_IP}" true 2>/dev/null; then
        log "stopping Ray on worker (${WORKER_IP})"
        worker_run "${ACTIVATE_CMD} && ray stop --force" >/dev/null 2>&1 || true
    else
        log "worker ${WORKER_IP} unreachable; skipping remote stop"
    fi
}

cmd_start() {
    local label="$1"
    parse_topology "${label}"
    local mask
    mask="$(device_mask "${GPUS}")"

    cmd_stop
    sleep 2

    local dashboard_flag="--include-dashboard=false"
    [ "${RAY_DASHBOARD}" = "1" ] && dashboard_flag="--include-dashboard=true"

    log "starting Ray head on ${HEAD_IP} with ${GPUS} GPU(s) (CUDA_VISIBLE_DEVICES=${mask})"
    CUDA_VISIBLE_DEVICES="${mask}" ray start --head \
        --node-ip-address="${HEAD_IP}" \
        --port="${RAY_PORT}" \
        --num-gpus="${GPUS}" \
        ${dashboard_flag} \
        --disable-usage-stats >/dev/null

    if [ "${NODES}" -eq 2 ]; then
        log "starting Ray worker on ${WORKER_IP} with ${GPUS} GPU(s)"
        worker_run "${ACTIVATE_CMD} && CUDA_VISIBLE_DEVICES=${mask} ray start \
            --address='${HEAD_IP}:${RAY_PORT}' \
            --num-gpus=${GPUS} \
            --disable-usage-stats" >/dev/null
    fi

    log "waiting for topology ${label} to register (up to ${CLUSTER_WAIT_SECONDS}s)"
    bench_python bench_common.py wait "${label}" --timeout "${CLUSTER_WAIT_SECONDS}" ||
        die "cluster did not reach topology ${label}; check 'ray status' on both nodes"
    log "cluster ready: ${label}"
}

cmd_status() {
    bench_python bench_common.py topology
}

[ $# -ge 1 ] || usage
case "$1" in
    start)  [ $# -eq 2 ] || usage; cmd_start "$2" ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    *)      usage ;;
esac
