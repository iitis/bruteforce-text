"""E7 - cost of returning partial results across one Ray node boundary.

``end_to_end`` measures the path used by a real worker task.  The timer starts before task
submission and stops only after ``ray.get`` has materialised every result in the driver.  A
payload-returning task is compared with a matched task which constructs the same payload but
returns ``None``.  There is no readiness wait before the timer: small Ray results can travel
inline with the task-completion reply, so waiting first would move that transfer outside the
measurement.

Every repetition is retained.  Local, remote and mixed placement are measured, and payload
sizes are measured rather than inferred from state counts.  The result is an aggregate return-
path diagnostic: it does not attempt to separate serialization, transport, storage and
materialization.

Usage (on an already configured two-node Ray cluster)::

    python benchmarks/exp_boundary_cost.py --topology 2x1
"""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

import bench_common as common


SCHEMA_VERSION = 4
DEFAULT_BATCH_SIZES = (16, 64)
DEFAULT_NUM_STATES = (1, 100, 1000)
DEFAULT_MAX_BATCH_BYTES = 16 * 1024 * 1024
DEFAULT_OPERATION_TIMEOUT_SECONDS = 300.0
DEFAULT_RUN_TIMEOUT_SECONDS = 20 * 60.0


def _payload(num_variables: int, num_states: int, seed: int = 0):
    """Build the same shape of SampleSet that a solver worker returns."""
    from dimod import SampleSet

    rng = np.random.default_rng(seed)
    samples = rng.choice([0, 1], size=(num_states, num_variables))
    energies = rng.normal(size=num_states)
    return SampleSet.from_samples(samples, "BINARY", energies)


def _summary(values: list[float], denominator: int) -> dict:
    """Summarise timings without discarding the raw observations stored alongside it."""
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    return {
        "count": int(array.size),
        "median_seconds": median,
        "min_seconds": float(np.min(array)),
        "max_seconds": float(np.max(array)),
        "median_per_item_seconds": median / denominator,
    }


def _positive_int(parser: argparse.ArgumentParser, name: str, value: int) -> None:
    if value <= 0:
        parser.error(f"{name} must be positive (got {value})")


def _effective_count(requested: int, payload_bytes: int, max_batch_bytes: int) -> int:
    """Keep a repetition bounded while always measuring at least one payload."""
    by_bytes = max(1, max_batch_bytes // max(1, payload_bytes))
    return min(requested, by_bytes)


def _run_id(value: str | None) -> str:
    run_id = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is None:
        raise ValueError(
            "--run-id may contain only ASCII letters, digits, dot, underscore and hyphen"
        )
    return run_id


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topology", default=None, help="expected cluster label, e.g. 2x1")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="end-to-end task batch sizes (default: 16 64)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="deprecated single-batch alias; overrides --batch-sizes when supplied",
    )
    parser.add_argument("--num-variables", type=int, default=60)
    parser.add_argument(
        "--num-states",
        type=int,
        nargs="+",
        default=list(DEFAULT_NUM_STATES),
        help="SampleSet state-count sweep (default: 1 100 1000)",
    )
    parser.add_argument(
        "--num-fixed-vars",
        type=int,
        default=3,
        help="fixed assignments represented in the common request payload",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--operation-timeout-seconds",
        type=float,
        default=DEFAULT_OPERATION_TIMEOUT_SECONDS,
        help="maximum wait for one Ray batch or fetch (default: 300)",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=DEFAULT_RUN_TIMEOUT_SECONDS,
        help="absolute deadline after connecting to Ray (default: 1200)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="exercise every E7 payload/placement path with two tasks and one repetition",
    )
    parser.add_argument(
        "--max-batch-bytes",
        type=int,
        default=DEFAULT_MAX_BATCH_BYTES,
        help="cap estimated payload bytes held by one batch (default: 16 MiB)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="optional unique output suffix; UTC timestamp with microseconds by default",
    )
    args = parser.parse_args()

    if args.num_tasks is not None:
        _positive_int(parser, "--num-tasks", args.num_tasks)
        batch_sizes = [args.num_tasks]
    else:
        batch_sizes = list(dict.fromkeys(args.batch_sizes))
    num_states_sweep = list(dict.fromkeys(args.num_states))
    for value in batch_sizes:
        _positive_int(parser, "--batch-sizes values", value)
    for value in num_states_sweep:
        _positive_int(parser, "--num-states values", value)
    _positive_int(parser, "--num-variables", args.num_variables)
    _positive_int(parser, "--repeats", args.repeats)
    _positive_int(parser, "--max-batch-bytes", args.max_batch_bytes)
    if not np.isfinite(args.operation_timeout_seconds) or args.operation_timeout_seconds <= 0:
        parser.error("--operation-timeout-seconds must be finite and positive")
    if not np.isfinite(args.run_timeout_seconds) or args.run_timeout_seconds <= 0:
        parser.error("--run-timeout-seconds must be finite and positive")
    if not 0 <= args.num_fixed_vars <= args.num_variables:
        parser.error("--num-fixed-vars must be between zero and --num-variables")

    if args.smoke:
        batch_sizes = [2]
        args.repeats = 1
        args.operation_timeout_seconds = min(args.operation_timeout_seconds, 30.0)
        args.run_timeout_seconds = min(args.run_timeout_seconds, 180.0)

    run_id = _run_id(args.run_id)

    import ray
    import ray.cloudpickle as ray_cloudpickle
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    run_started = perf_counter()

    def ray_get(refs, stage: str):
        elapsed = perf_counter() - run_started
        remaining = args.run_timeout_seconds - elapsed
        if remaining <= 0:
            raise TimeoutError(
                f"E7 exceeded its {args.run_timeout_seconds:g}s run deadline before {stage}"
            )
        timeout = min(args.operation_timeout_seconds, remaining)
        try:
            return ray.get(refs, timeout=timeout)
        except ray.exceptions.GetTimeoutError as error:
            raise TimeoutError(
                f"E7 timed out after {timeout:.1f}s during {stage}"
            ) from error

    topology = common.require_topology(args.topology)
    if topology["num_nodes"] < 2:
        raise SystemExit(
            f"This experiment needs at least two nodes; the cluster reports "
            f"{topology['label']}. Start a 2xM cluster with scripts/ray_cluster.sh."
        )

    controller_node_id = ray.get_runtime_context().get_node_id()
    topology_node_ids = {node["node_id"] for node in topology["nodes"]}
    if controller_node_id not in topology_node_ids:
        raise SystemExit(
            "The driver is not running on one of the GPU-carrying nodes described by the "
            "requested topology; run E7 from the Ray head node used by the benchmark."
        )
    remote_node_ids = sorted(topology_node_ids - {controller_node_id})
    if not remote_node_ids:
        raise SystemExit("Could not identify a node other than the controller's.")
    remote_node_id = remote_node_ids[0]

    mode = "smoke" if args.smoke else "production"
    output_name = f"{topology['label']}_{mode}_{run_id}"
    output_path = common.RESULTS_DIR / "boundary_cost" / f"{output_name}.json"
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing E7 result {output_path}; choose another --run-id."
        )

    def pin(node_id: str):
        return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

    # The baseline deliberately constructs the same payload.  Its only difference from the
    # payload task is whether that object crosses the task return boundary.
    @ray.remote(num_cpus=1)
    def _discard_task(request_blob, num_variables, num_states, seed):
        _ = len(request_blob)
        payload = _payload(num_variables, num_states, seed)
        _ = len(payload)
        return None

    @ray.remote(num_cpus=1)
    def _payload_task(request_blob, num_variables, num_states, seed):
        _ = len(request_blob)
        return _payload(num_variables, num_states, seed)

    @ray.remote(num_cpus=0)
    def _node_environment():
        import os
        import platform
        import subprocess
        import sys
        from importlib.metadata import PackageNotFoundError, version

        packages = {}
        for package in ("dimod", "numpy", "ray"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None

        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        gpus = []
        if gpu_query.returncode == 0:
            for line in gpu_query.stdout.splitlines():
                fields = [field.strip() for field in line.split(",", 2)]
                if len(fields) == 3:
                    gpus.append(
                        {"name": fields[0], "uuid": fields[1], "driver": fields[2]}
                    )

        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "hostname": platform.node(),
                "cpu_count": os.cpu_count(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpus": gpus,
                "nvidia_smi_error": (
                    gpu_query.stderr.strip() if gpu_query.returncode != 0 else None
                ),
                "packages": packages,
            },
        }

    # A deterministic dense-QUBO-shaped request makes input scheduling realistic.  Both task
    # variants receive the same immutable byte string, so it cancels in their paired contrast.
    request_rng = np.random.default_rng(20240801)
    upper_i, upper_j = np.triu_indices(args.num_variables)
    coefficients = request_rng.uniform(-1.0, 1.0, size=upper_i.size)
    request = {
        "num_variables": args.num_variables,
        "quadratic": [
            (int(i), int(j), float(value))
            for i, j, value in zip(upper_i, upper_j, coefficients)
        ],
        "fixed_variables": {i: i % 2 for i in range(args.num_fixed_vars)},
    }
    request_blob = ray_cloudpickle.dumps(request)
    request_descriptor = {
        "kind": "deterministic dense-QUBO-shaped serialized request",
        "serialized_bytes": len(request_blob),
        "sha256": hashlib.sha256(request_blob).hexdigest(),
        "num_quadratic_terms": len(request["quadratic"]),
        "num_fixed_variables": len(request["fixed_variables"]),
    }

    node_environments = {}
    for label, node_id in (("local", controller_node_id), ("remote", remote_node_id)):
        node_environments[label] = ray_get(
            _node_environment.options(scheduling_strategy=pin(node_id)).remote(),
            f"collecting the {label} node environment",
        )

    print(
        f"controller {controller_node_id[:12]}..., remote {remote_node_id[:12]}..., "
        f"run_id={run_id}, mode={mode}",
        flush=True,
    )

    placements = {
        "local": lambda count: [controller_node_id] * count,
        "remote": lambda count: [remote_node_id] * count,
        "mixed": lambda count: [
            controller_node_id if index % 2 == 0 else remote_node_id
            for index in range(count)
        ],
    }

    def timed_task_batch(task, node_ids, num_states, seed_base):
        start = perf_counter()
        refs = [
            task.options(scheduling_strategy=pin(node_id)).remote(
                request_blob, args.num_variables, num_states, seed_base + index
            )
            for index, node_id in enumerate(node_ids)
        ]
        values = ray_get(refs, "materializing an end-to-end task batch")
        elapsed = perf_counter() - start
        del refs, values
        return elapsed

    payload_descriptors = {}
    end_to_end = {}

    for num_states in num_states_sweep:
        template = _payload(args.num_variables, num_states, seed=9191 + num_states)
        serialized_payload_bytes = len(ray_cloudpickle.dumps(template))
        payload_descriptors[str(num_states)] = {
            "num_states": num_states,
            "num_variables": args.num_variables,
            "driver_cloudpickle_bytes": serialized_payload_bytes,
            "sample_dtype": str(template.record.sample.dtype),
            "energy_dtype": str(template.record.energy.dtype),
        }
        del template

        # Warm both function variants on both nodes outside all recorded intervals.
        print(f"  states={num_states:<4d} warm-up starting", flush=True)
        for task_name, task in (("discard", _discard_task), ("payload", _payload_task)):
            ray_get(
                [
                    task.options(scheduling_strategy=pin(node_id)).remote(
                        request_blob, args.num_variables, num_states, 8000 + index
                    )
                    for index, node_id in enumerate(
                        (controller_node_id, remote_node_id)
                    )
                ],
                f"warming {task_name} tasks for {num_states} states",
            )

        state_end_to_end = {}
        for requested_batch_size in batch_sizes:
            batch_size = _effective_count(
                requested_batch_size,
                serialized_payload_bytes,
                args.max_batch_bytes,
            )
            runs_by_placement = {name: [] for name in placements}
            print(
                f"  states={num_states:<4d} batch={batch_size:<4d} "
                f"(requested {requested_batch_size}) end-to-end starting",
                flush=True,
            )

            for repeat in range(args.repeats):
                print(
                    f"    end-to-end repeat {repeat + 1}/{args.repeats}",
                    flush=True,
                )
                placement_order = list(placements)
                shift = repeat % len(placement_order)
                placement_order = placement_order[shift:] + placement_order[:shift]
                for placement_name in placement_order:
                    node_ids = placements[placement_name](batch_size)
                    task_order = (
                        ("discard", "payload")
                        if (repeat + list(placements).index(placement_name)) % 2 == 0
                        else ("payload", "discard")
                    )
                    seed_base = (
                        num_states * 10_000_000
                        + requested_batch_size * 10_000
                        + repeat * batch_size
                    )
                    seconds = {}
                    for task_name in task_order:
                        print(
                            f"      placement={placement_name:<6s} task={task_name}",
                            flush=True,
                        )
                        task = _discard_task if task_name == "discard" else _payload_task
                        seconds[task_name] = timed_task_batch(
                            task, node_ids, num_states, seed_base
                        )
                    runs_by_placement[placement_name].append(
                        {
                            "repeat": repeat,
                            "task_order": list(task_order),
                            "discard_seconds": seconds["discard"],
                            "payload_seconds": seconds["payload"],
                            "payload_minus_discard_seconds": (
                                seconds["payload"] - seconds["discard"]
                            ),
                            "payload_minus_discard_per_task_seconds": (
                                seconds["payload"] - seconds["discard"]
                            )
                            / batch_size,
                        }
                    )

            summaries = {}
            for placement_name, runs in runs_by_placement.items():
                discard_values = [run["discard_seconds"] for run in runs]
                payload_values = [run["payload_seconds"] for run in runs]
                delta_values = [run["payload_minus_discard_seconds"] for run in runs]
                summaries[placement_name] = {
                    "discard": _summary(discard_values, batch_size),
                    "payload": _summary(payload_values, batch_size),
                    "payload_minus_discard": _summary(delta_values, batch_size),
                }

            paired_comparisons = []
            for repeat in range(args.repeats):
                by_name = {
                    name: runs_by_placement[name][repeat] for name in placements
                }
                paired_comparisons.append(
                    {
                        "repeat": repeat,
                        "remote_minus_local_payload_seconds": (
                            by_name["remote"]["payload_seconds"]
                            - by_name["local"]["payload_seconds"]
                        ),
                        "remote_minus_local_discard_seconds": (
                            by_name["remote"]["discard_seconds"]
                            - by_name["local"]["discard_seconds"]
                        ),
                        "remote_minus_local_return_delta_seconds": (
                            by_name["remote"]["payload_minus_discard_seconds"]
                            - by_name["local"]["payload_minus_discard_seconds"]
                        ),
                    }
                )

            placement_task_counts = {
                name: {
                    "local": node_ids.count(controller_node_id),
                    "remote": node_ids.count(remote_node_id),
                }
                for name, node_ids in (
                    (name, factory(batch_size)) for name, factory in placements.items()
                )
            }
            state_end_to_end[str(requested_batch_size)] = {
                "requested_batch_size": requested_batch_size,
                "effective_batch_size": batch_size,
                "payload_byte_cap_applied": batch_size < requested_batch_size,
                "placement_task_counts": placement_task_counts,
                "runs_by_placement": runs_by_placement,
                "summaries": summaries,
                "paired_comparisons": paired_comparisons,
            }
            print(
                f"  states={num_states:<4d} batch={batch_size:<4d} "
                f"(requested {requested_batch_size}) end-to-end complete",
                flush=True,
            )
        end_to_end[str(num_states)] = state_end_to_end

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "mode": mode,
        "topology": topology,
        "node_roles": {
            "controller_node_id": controller_node_id,
            "remote_node_id": remote_node_id,
        },
        "node_environments": node_environments,
        "methodology": {
            "end_to_end_timer": (
                "perf_counter immediately before submitting the full task batch through "
                "completion of ray.get(refs); no ray.wait precedes the timer"
            ),
            "end_to_end_baseline": (
                "matched task receives the same request and constructs the same SampleSet, "
                "but returns None"
            ),
            "interpretation": (
                "payload-minus-discard contrasts include serialization, ordinary Ray delivery "
                "and materialization; the remote-minus-local difference between those "
                "contrasts is the observed node-boundary effect under this two-node load, not "
                "an isolated network term or a prediction of larger-node saturation"
            ),
        },
        "request": request_descriptor,
        "num_variables": args.num_variables,
        "num_states_sweep": num_states_sweep,
        "batch_sizes_requested": batch_sizes,
        "max_batch_bytes": args.max_batch_bytes,
        "repeats": args.repeats,
        "operation_timeout_seconds": args.operation_timeout_seconds,
        "run_timeout_seconds": args.run_timeout_seconds,
        "provenance": {
            "driver": str(Path(__file__).resolve().relative_to(common.BENCHMARKS_DIR)),
            "driver_sha256": _sha256_file(Path(__file__)),
            "shared_helpers": str(
                Path(common.__file__).resolve().relative_to(common.BENCHMARKS_DIR)
            ),
            "shared_helpers_sha256": _sha256_file(Path(common.__file__)),
        },
        "payload_descriptors": payload_descriptors,
        "end_to_end": end_to_end,
    }
    written = common.write_result("boundary_cost", output_name, result)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
