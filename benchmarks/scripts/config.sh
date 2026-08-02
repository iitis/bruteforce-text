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
# Command that makes ${CONDA_ENV} active in a fresh shell. Adjust if conda lives elsewhere
# or if you use a plain virtualenv (then set it to e.g. 'source /path/venv/bin/activate').
: "${ACTIVATE_CMD:=conda activate ${CONDA_ENV}}"
: "${PYTHON:=python}"

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
: "${CONTROLLER_MAX_K:=16}"
# Skip an experiment point whose result file already exists.
: "${SKIP_EXISTING:=1}"

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
