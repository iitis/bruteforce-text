# Shared configuration for the benchmark launchers. Sourced, not executed.
#
# Every value can be overridden from the environment, e.g.
#   WORKER_IP=10.40.40.107 ./scripts/run_e1_strong_scaling.sh
#
# All scripts assume they are started on the head node.

# --- cluster ---------------------------------------------------------------------------
: "${HEAD_IP:=10.40.40.105}"
: "${WORKER_IP:=10.40.40.106}"
: "${RAY_PORT:=6379}"
: "${RAY_DASHBOARD:=0}"          # 1 to enable the Ray dashboard
: "${CLUSTER_WAIT_SECONDS:=180}" # how long to wait for all GPUs to register

# Maximum number of GPUs per node. Topologies are built by masking down from this.
: "${GPUS_PER_NODE:=4}"

# --- software --------------------------------------------------------------------------
: "${CONDA_ENV:=omnisolver-bruteforce-bench}"
# Command that makes ${CONDA_ENV} active in a fresh, *non-interactive* shell. Plain
# "conda activate" is not enough there: unless "conda init" has been run for that shell, conda
# refuses with "Run 'conda init' before 'conda activate'". Sourcing conda's profile script
# first works whether or not init has been run, and it also puts the environment's own "ray"
# on PATH, which ray_cluster.sh relies on. The single quotes keep $(conda info --base)
# unevaluated so that it resolves on whichever machine runs the command.
# Override for a plain virtualenv, e.g. ACTIVATE_CMD='source /path/venv/bin/activate'.
# Base prefix of the conda installation. Leave empty to auto-detect; set it if a node keeps
# conda somewhere unusual, e.g. CONDA_BASE=/opt/miniconda3.
: "${CONDA_BASE:=}"
if [ -z "${ACTIVATE_CMD:-}" ]; then
    # Tries, in order: an explicit CONDA_BASE, conda on PATH, then the usual prefixes. Needed
    # because a non-interactive ssh shell often has neither the conda function nor conda on
    # PATH, even on a node where the interactive shell has both.
    # shellcheck disable=SC2016
    ACTIVATE_CMD='__b="'"${CONDA_BASE}"'"; [ -z "$__b" ] && command -v conda >/dev/null 2>&1 && __b="$(conda info --base)"; for __c in "$__b" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/anaconda3" /opt/miniconda3 /opt/conda /usr/local/miniconda3; do [ -n "$__c" ] && [ -f "$__c/etc/profile.d/conda.sh" ] && { . "$__c/etc/profile.d/conda.sh"; break; }; done; command -v conda >/dev/null 2>&1 || { echo "conda not found; set CONDA_BASE or ACTIVATE_CMD" >&2; exit 1; }; conda activate '"${CONDA_ENV}"
fi
# Interpreter and Ray CLI to use on this (head) node. Resolved once, through ACTIVATE_CMD, to
# absolute paths inside ${CONDA_ENV}. This makes the launchers independent of which environment
# the calling shell happens to have active: running them from (base) would otherwise pick up
# base's python, which has neither the plugin nor ray, and fail deep inside an experiment.
# Set PYTHON/RAY explicitly to override.
if [ -z "${PYTHON:-}" ]; then
    PYTHON="$(bash -lc "${ACTIVATE_CMD} >/dev/null 2>&1 && command -v python" 2>/dev/null)"
    [ -n "${PYTHON}" ] || PYTHON=python
fi
if [ -z "${RAY:-}" ]; then
    RAY="$(bash -lc "${ACTIVATE_CMD} >/dev/null 2>&1 && command -v ray" 2>/dev/null)"
    [ -n "${RAY}" ] || RAY=ray
fi

# --- paths -----------------------------------------------------------------------------
# Resolved from the location of this file, so the scripts work from any directory.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"
REPO_DIR="$(cd "${BENCHMARKS_DIR}/.." && pwd)"
# Where this repository lives on the worker node. Only needed if you want the worker to see
# the same instance files; the solver itself only needs the installed package.
: "${WORKER_REPO_DIR:=${REPO_DIR}}"
: "${LOG_DIR:=${BENCHMARKS_DIR}/logs}"

# --- ssh -------------------------------------------------------------------------------
: "${SSH:=ssh -o BatchMode=yes -o ConnectTimeout=10}"

# --- experiment defaults ---------------------------------------------------------------
# Topologies exercised by the scaling experiments, as <nodes>x<gpus per node>.
: "${TOPOLOGIES:=1x1 1x2 2x1 1x4 2x2 2x4}"
: "${STRONG_SCALING_SIZE:=50}"
: "${WEAK_SCALING_BASE_SIZE:=50}"
: "${PRECISION_SIZES:=40 42 44 46}"
: "${QUBO_SIZES:=40 42 44 46 48 50}"
: "${FAMILIES_SIZE:=44}"
: "${FAMILIES_REPLICAS:=5}"
: "${WISHART_SIZE:=40}"
: "${WISHART_REPLICAS:=20}"
: "${WISHART_ALPHAS:=0.2 0.25 0.3 0.35 0.4 0.45 0.5}"
: "${WISHART_ENSEMBLES:=planted unplanted}"
: "${WISHART_TTS_ALPHAS:=0.2}"
: "${WISHART_TTS_REPEATS:=100}"
: "${CONTROLLER_MAX_K:=16}"
# Skip an experiment point whose result file already exists.
: "${SKIP_EXISTING:=1}"

# Stream experiment output line by line: the launchers pipe through tee, and Python would
# otherwise block-buffer its progress lines into multi-minute bursts.
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"

# --- helpers ---------------------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# Run a command on the worker node inside a login shell, so that conda is on PATH.
worker_run() {
    # shellcheck disable=SC2086
    ${SSH} "${WORKER_IP}" "bash -lc $(printf '%q' "$*")"
}

# Run a python command from the benchmarks directory on this (head) node.
bench_python() {
    (cd "${BENCHMARKS_DIR}" && "${PYTHON}" "$@")
}

# True if $1 (a results-relative path) already exists and skipping is enabled.
result_exists() {
    [ "${SKIP_EXISTING}" = "1" ] && [ -f "${BENCHMARKS_DIR}/results/$1" ]
}
