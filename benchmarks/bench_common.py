"""Shared helpers for the benchmark and verification experiments.

Collects the three things every experiment needs and that the original benchmark script did
not record: a description of the software and hardware it ran on, a description of the Ray
cluster topology it ran against, and a deterministic per-size instance generator.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import typing
from pathlib import Path

import numpy as np

BENCHMARKS_DIR = Path(__file__).resolve().parent
INSTANCES_DIR = BENCHMARKS_DIR / "instances"
RESULTS_DIR = BENCHMARKS_DIR / "results"

#: Solver parameters used for every runtime measurement reported in the manuscript. Kept in
#: one place so that the scaling, precision and QUBO experiments are directly comparable.
KERNEL_PARAMS = {
    "suffix_size": 23,
    "grid_size": 8192,
    "block_size": 1024,
    "num_steps_per_kernel": 8192,
    "partial_diff_buffer_depth": 10,
}

#: Seed of the instance generator. Instances are drawn per size (see
#: :func:`generate_instance`), so a given (size, family, seed) always yields the same file
#: regardless of which other sizes are generated alongside it.
DEFAULT_SEED = 42

INSTANCE_FAMILIES = ("uniform", "bimodal", "gaussian", "sparse")


# --------------------------------------------------------------------------------------
# Environment descriptors
# --------------------------------------------------------------------------------------
def _package_version(name: str) -> typing.Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _cuda_versions() -> dict:
    versions: dict = {"runtime": None, "driver": None, "devices": []}
    try:
        from numba import cuda

        if cuda.is_available():
            major, minor = cuda.runtime.get_version()
            versions["runtime"] = f"{major}.{minor}"
            versions["devices"] = [
                {
                    "name": device.name.decode() if isinstance(device.name, bytes) else device.name,
                    "compute_capability": "%d.%d" % device.compute_capability,
                }
                for device in cuda.gpus
            ]
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            versions["driver"] = out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return versions


def environment_descriptors() -> dict:
    """Everything needed to interpret a timing later on.

    The absence of these descriptors in the original result files is why the CUDA Toolkit
    version used for the published measurements had to be reconstructed by hand.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "cuda": _cuda_versions(),
        "packages": {
            name: _package_version(name)
            for name in (
                "omnisolver",
                "omnisolver-bruteforce",
                "dimod",
                "numpy",
                "numba",
                "ray",
            )
        },
    }


# --------------------------------------------------------------------------------------
# Ray cluster topology
# --------------------------------------------------------------------------------------
class TopologyError(RuntimeError):
    """Raised when the running Ray cluster does not match the requested topology."""


def detect_topology() -> dict:
    """Describe the GPU layout of the Ray cluster this process is connected to.

    :returns: a dictionary with the number of GPU-carrying nodes, the GPUs per node, the
        total GPU count and a short label such as ``"2x4"`` (2 nodes, 4 GPUs each).
    """
    import ray

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    per_node = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        gpus = int(node.get("Resources", {}).get("GPU", 0))
        if gpus:
            per_node.append({"node_id": node["NodeID"], "ip": node.get("NodeManagerAddress"), "gpus": gpus})

    counts = sorted({entry["gpus"] for entry in per_node})
    homogeneous = len(counts) <= 1
    gpus_per_node = counts[0] if counts else 0
    total = sum(entry["gpus"] for entry in per_node)

    return {
        "num_nodes": len(per_node),
        "gpus_per_node": gpus_per_node,
        "total_gpus": total,
        "homogeneous": homogeneous,
        "label": f"{len(per_node)}x{gpus_per_node}",
        "nodes": per_node,
    }


def require_topology(expected_label: typing.Optional[str]) -> dict:
    """Detect the topology and, if a label was requested, check that it matches.

    Ray cannot rearrange an already running cluster, so the topology is a property of how
    ``ray start`` was invoked on each node. This makes the experiment fail loudly instead of
    silently mislabelling its output.
    """
    topology = detect_topology()
    if topology["total_gpus"] == 0:
        raise TopologyError("The Ray cluster reports no GPUs; start it with --num-gpus=<n>.")
    if not topology["homogeneous"]:
        raise TopologyError(
            f"Nodes carry different GPU counts ({topology['nodes']}); the scaling experiments "
            "assume a homogeneous cluster."
        )
    if expected_label is not None and topology["label"] != expected_label:
        raise TopologyError(
            f"Requested topology {expected_label} but the cluster reports {topology['label']} "
            f"({topology['num_nodes']} node(s), {topology['gpus_per_node']} GPU(s) each)."
        )
    return topology


def num_fixed_vars_for(num_gpus: int) -> int:
    """Number of fixed variables placing exactly one subproblem on each GPU."""
    k = int(round(np.log2(num_gpus)))
    if 2**k != num_gpus:
        raise ValueError(
            f"The scaling experiments require a power-of-two GPU count, but the cluster has "
            f"{num_gpus}."
        )
    return k


# --------------------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------------------
def instance_path(num_variables: int, family: str = "uniform") -> Path:
    """Location of an instance file. The historical ``uniform`` family keeps a flat layout."""
    if family == "uniform":
        return INSTANCES_DIR / f"{num_variables}.txt"
    return INSTANCES_DIR / family / f"{num_variables}.txt"


def _draw_couplings(num_variables: int, family: str, rng: np.random.Generator) -> np.ndarray:
    if family == "uniform":
        return 2 * (rng.random((num_variables, num_variables)) - 0.5)
    if family == "bimodal":
        return rng.choice([-1.0, 1.0], size=(num_variables, num_variables))
    if family == "gaussian":
        return rng.normal(0.0, 1.0, size=(num_variables, num_variables))
    if family == "sparse":
        dense = 2 * (rng.random((num_variables, num_variables)) - 0.5)
        mask = rng.random((num_variables, num_variables)) < 0.2
        return dense * mask
    raise ValueError(f"Unknown instance family {family!r}; expected one of {INSTANCE_FAMILIES}.")


def generate_instance(
    num_variables: int,
    family: str = "uniform",
    seed: int = DEFAULT_SEED,
    replica: int = 0,
    overwrite: bool = False,
) -> Path:
    """Write an instance file unless it already exists, and return its path.

    Unlike the sweep-based generator of :mod:`bf.py`, the random stream is seeded from
    ``(seed, num_variables, family, replica)``, so the content of a given file does not depend
    on which other sizes happen to be generated in the same run. Existing files are never
    rewritten unless ``overwrite`` is set, because the instances behind the published results
    are part of the record.
    """
    path = instance_path(num_variables, family) if replica == 0 else (
        INSTANCES_DIR / family / f"{num_variables}_r{replica}.txt"
    )
    if path.is_file() and not overwrite:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    entropy = [seed, num_variables, replica, sum(ord(c) for c in family)]
    rng = np.random.default_rng(entropy)
    couplings = _draw_couplings(num_variables, family, rng)
    with open(path, "w") as fd:
        for i in range(num_variables):
            for j in range(i, num_variables):
                fd.write(f"{i} {j} {couplings[i, j]}\n")
    return path


def load_bqm(path, vartype: str = "SPIN"):
    """Read an instance into a ``dimod`` binary quadratic model."""
    from dimod.serialization import coo

    with open(path) as fd:
        return coo.load(fd, vartype=vartype)


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_result(experiment: str, name: str, payload: dict) -> Path:
    """Store one experiment result, stamped with the environment it was produced in."""
    directory = RESULTS_DIR / experiment
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    enriched = dict(payload)
    enriched.setdefault("experiment", experiment)
    enriched["environment"] = environment_descriptors()
    with open(path, "w") as fd:
        json.dump(enriched, fd, indent=4, cls=NumpyEncoder)
    return path


def load_results(experiment: str) -> list:
    directory = RESULTS_DIR / experiment
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        with open(path) as fd:
            out.append(json.load(fd))
    return out


# --------------------------------------------------------------------------------------
# Small command line interface, used by the launcher scripts
# --------------------------------------------------------------------------------------
def _cli_topology() -> int:
    print(json.dumps(detect_topology(), indent=2))
    return 0


def _cli_wait(expected_label: str, timeout_seconds: float) -> int:
    """Block until the Ray cluster reports the expected topology.

    ``ray start`` returns as soon as the local daemon is up, so a worker node may not have
    registered its GPUs yet when the next command runs. Polling for the expected label
    removes that race.
    """
    from time import perf_counter, sleep

    deadline = perf_counter() + timeout_seconds
    last = None
    while perf_counter() < deadline:
        try:
            topology = detect_topology()
            last = topology["label"]
            if last == expected_label:
                print(f"cluster ready: {last} ({topology['total_gpus']} GPU(s))")
                return 0
        except Exception as error:  # cluster not up yet
            last = f"unavailable ({error})"
        sleep(2)
    print(
        f"timed out after {timeout_seconds:g}s waiting for topology {expected_label}; "
        f"last seen: {last}",
        file=sys.stderr,
    )
    return 1


def _cli_environment() -> int:
    print(json.dumps(environment_descriptors(), indent=2))
    return 0


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cluster and environment introspection.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("topology", help="print the detected Ray GPU topology as JSON")
    sub.add_parser("environment", help="print the environment descriptors as JSON")
    wait = sub.add_parser("wait", help="block until the cluster reports a given topology")
    wait.add_argument("label", help="expected topology label, e.g. 2x4")
    wait.add_argument("--timeout", type=float, default=120.0)

    args = parser.parse_args(argv)
    if args.command == "topology":
        return _cli_topology()
    if args.command == "environment":
        return _cli_environment()
    return _cli_wait(args.label, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
