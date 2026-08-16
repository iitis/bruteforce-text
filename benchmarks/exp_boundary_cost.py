"""E7 - cost of returning partial results across one Ray node boundary.

This experiment has two deliberately separate parts.

``end_to_end`` measures the path used by a real worker task.  The timer starts before task
submission and stops only after ``ray.get`` has materialised every result in the driver.  A
payload-returning task is compared with a matched task which constructs the same payload but
returns ``None``.  There is no readiness wait before the timer: small Ray results can travel
inline with the task-completion reply, so waiting first would move that transfer outside the
measurement.

``nested_ref_fetch`` isolates a different mechanism.  A task pinned to the selected node calls
``ray.put`` for each payload and returns a *nested* ObjectRef.  The driver waits for those inner
references with ``fetch_local=False``, then times their first and second ``ray.get``.  Ray may
keep small values in worker memory rather than Plasma, so the driver records the storage class
reported by Ray instead of assuming that every ``ray.put`` value is object-store-backed.  For a
payload that Ray reports on the producer's object-store node, the first remote get includes the
store-to-store transfer and the second is a warm-local control.  These measurements are kept
separate from ordinary task returns.

Both parts retain every repetition.  Local, remote and mixed placement are measured for the
end-to-end path, and payload sizes are measured rather than inferred from state counts.

Usage (on an already configured two-node Ray cluster)::

    python benchmarks/exp_boundary_cost.py --topology 2x1
"""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

import bench_common as common


SCHEMA_VERSION = 3
DEFAULT_BATCH_SIZES = (32, 256)
DEFAULT_NUM_STATES = (1, 100, 1000)
DEFAULT_NUM_OBJECTS = 256
DEFAULT_MAX_BATCH_BYTES = 16 * 1024 * 1024


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topology", default=None, help="expected cluster label, e.g. 2x1")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="end-to-end task batch sizes (default: 32 256)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="deprecated single-batch alias; overrides --batch-sizes when supplied",
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=DEFAULT_NUM_OBJECTS,
        help="requested objects per forced-fetch repetition (default: 256)",
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
    _positive_int(parser, "--num-objects", args.num_objects)
    _positive_int(parser, "--num-variables", args.num_variables)
    _positive_int(parser, "--repeats", args.repeats)
    _positive_int(parser, "--max-batch-bytes", args.max_batch_bytes)
    if not 0 <= args.num_fixed_vars <= args.num_variables:
        parser.error("--num-fixed-vars must be between zero and --num-variables")

    run_id = _run_id(args.run_id)

    import ray
    import ray.cloudpickle as ray_cloudpickle
    from ray.experimental import get_object_locations
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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

    output_name = f"{topology['label']}_{run_id}"
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

    # Returning a container prevents the inner reference from becoming the task's ordinary
    # top-level value.  Waiting on that inner reference with fetch_local=False does not pull
    # its bytes to the driver before the fetch timer starts.
    @ray.remote(num_cpus=1)
    def _put_payload_task(num_variables, num_states, seed):
        payload_ref = ray.put(_payload(num_variables, num_states, seed))
        return {"payload_ref": payload_ref}

    @ray.remote(num_cpus=0)
    def _node_environment():
        import os
        import platform
        import sys
        from importlib.metadata import PackageNotFoundError, version

        packages = {}
        for package in ("dimod", "numpy", "ray"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None

        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "hostname": platform.node(),
                "cpu_count": os.cpu_count(),
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
        node_environments[label] = ray.get(
            _node_environment.options(scheduling_strategy=pin(node_id)).remote()
        )

    print(
        f"controller {controller_node_id[:12]}..., remote {remote_node_id[:12]}..., "
        f"run_id={run_id}"
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
        values = ray.get(refs)
        elapsed = perf_counter() - start
        del refs, values
        return elapsed

    payload_descriptors = {}
    end_to_end = {}
    nested_ref_fetch = {}

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
        for task in (_discard_task, _payload_task):
            ray.get(
                [
                    task.options(scheduling_strategy=pin(node_id)).remote(
                        request_blob, args.num_variables, num_states, 8000 + index
                    )
                    for index, node_id in enumerate(
                        (controller_node_id, remote_node_id)
                    )
                ]
            )

        state_end_to_end = {}
        for requested_batch_size in batch_sizes:
            batch_size = _effective_count(
                requested_batch_size,
                serialized_payload_bytes,
                args.max_batch_bytes,
            )
            runs_by_placement = {name: [] for name in placements}

            for repeat in range(args.repeats):
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
                f"(requested {requested_batch_size}) end-to-end complete"
            )
        end_to_end[str(num_states)] = state_end_to_end

        object_count = _effective_count(
            args.num_objects, serialized_payload_bytes, args.max_batch_bytes
        )

        def create_inner_refs(node_id, seed_base):
            outer_refs = [
                _put_payload_task.options(scheduling_strategy=pin(node_id)).remote(
                    args.num_variables, num_states, seed_base + index
                )
                for index in range(object_count)
            ]
            wrappers = ray.get(outer_refs)
            inner_refs = [wrapper["payload_ref"] for wrapper in wrappers]
            ready, remaining = ray.wait(
                inner_refs, num_returns=len(inner_refs), fetch_local=False
            )
            if remaining:
                raise RuntimeError(
                    "ray.wait returned before all nested-reference objects were ready"
                )
            return ready

        def location_snapshot(refs, expected_node_id):
            location_info = get_object_locations(refs)
            sizes = []
            expected_count = 0
            objects_with_reported_locations = 0
            node_id_counts = {}
            for ref in refs:
                info = location_info.get(ref, {})
                sizes.append(int(info.get("object_size", 0)))
                node_ids = [str(node_id) for node_id in info.get("node_ids", [])]
                if node_ids:
                    objects_with_reported_locations += 1
                if expected_node_id in node_ids:
                    expected_count += 1
                for node_id in node_ids:
                    node_id_counts[node_id] = node_id_counts.get(node_id, 0) + 1
            unexpected_node_ids = sorted(set(node_id_counts) - {expected_node_id})
            if expected_count == len(refs) and not unexpected_node_ids:
                storage_class = "object_store_on_expected_node"
            elif objects_with_reported_locations == 0:
                # Ray's direct-call/worker-memory values do not have Plasma node locations.
                storage_class = "not_reported_in_object_store"
            else:
                storage_class = "mixed_or_unexpected"
            return {
                "num_objects": len(refs),
                "objects_with_reported_locations": objects_with_reported_locations,
                "objects_on_expected_node": expected_count,
                "all_on_expected_node": expected_count == len(refs),
                "storage_class": storage_class,
                "node_id_counts": node_id_counts,
                "unexpected_node_ids": unexpected_node_ids,
                "object_size_bytes": {
                    "min": min(sizes),
                    "median": float(np.median(sizes)),
                    "max": max(sizes),
                },
            }

        def timed_nested_fetch(node_id, seed_base):
            refs = create_inner_refs(node_id, seed_base)
            before = location_snapshot(refs, node_id)
            if before["storage_class"] == "mixed_or_unexpected":
                raise RuntimeError(
                    "Nested-reference location validation failed before fetch: "
                    f"expected either no Plasma locations or all objects on {node_id}; "
                    f"observed {before}"
                )
            start = perf_counter()
            values = ray.get(refs)
            cold_seconds = perf_counter() - start
            start = perf_counter()
            warm_values = ray.get(refs)
            warm_seconds = perf_counter() - start
            del refs, values, warm_values
            return {
                "cold_seconds": cold_seconds,
                "warm_seconds": warm_seconds,
                "cold_per_object_seconds": cold_seconds / object_count,
                "warm_per_object_seconds": warm_seconds / object_count,
                "locations_before_fetch": before,
            }

        forced_runs = []
        for repeat in range(args.repeats):
            location_order = (
                ("local", "remote") if repeat % 2 == 0 else ("remote", "local")
            )
            observations = {}
            for label in location_order:
                node_id = controller_node_id if label == "local" else remote_node_id
                seed_base = num_states * 20_000_000 + repeat * object_count
                observations[label] = timed_nested_fetch(node_id, seed_base)
            forced_runs.append(
                {
                    "repeat": repeat,
                    "location_order": list(location_order),
                    "local": observations["local"],
                    "remote": observations["remote"],
                    "remote_minus_local_cold_seconds": (
                        observations["remote"]["cold_seconds"]
                        - observations["local"]["cold_seconds"]
                    ),
                    "remote_minus_local_cold_per_object_seconds": (
                        observations["remote"]["cold_seconds"]
                        - observations["local"]["cold_seconds"]
                    )
                    / object_count,
                }
            )

        local_cold = [run["local"]["cold_seconds"] for run in forced_runs]
        local_warm = [run["local"]["warm_seconds"] for run in forced_runs]
        remote_cold = [run["remote"]["cold_seconds"] for run in forced_runs]
        remote_warm = [run["remote"]["warm_seconds"] for run in forced_runs]
        remote_minus_local = [
            run["remote_minus_local_cold_seconds"] for run in forced_runs
        ]
        nested_ref_fetch[str(num_states)] = {
            "requested_object_count": args.num_objects,
            "effective_object_count": object_count,
            "payload_byte_cap_applied": object_count < args.num_objects,
            "runs": forced_runs,
            "summaries": {
                "local_cold": _summary(local_cold, object_count),
                "local_warm": _summary(local_warm, object_count),
                "remote_cold": _summary(remote_cold, object_count),
                "remote_warm": _summary(remote_warm, object_count),
                "remote_minus_local_cold": _summary(
                    remote_minus_local, object_count
                ),
            },
        }
        print(
            f"  states={num_states:<4d} objects={object_count:<4d} "
            f"(requested {args.num_objects}) nested-reference fetch complete"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
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
            "nested_ref_fetch_timer": (
                "workers call ray.put and return nested ObjectRefs; driver calls "
                "ray.wait(inner_refs, fetch_local=False) before timing the first ray.get; "
                "second ray.get is the warm-local control; Ray-reported object locations "
                "distinguish Plasma-backed values from smaller values retained elsewhere"
            ),
            "interpretation": (
                "end-to-end contrasts include scheduling, return serialization and inline "
                "or object-store delivery; nested-reference remote-minus-local contrasts "
                "measure the additional first-fetch path under this two-node load, and only "
                "rows reported on the producer node are identified as object-store transfers; "
                "neither path establishes scaling or saturation at larger node counts"
            ),
        },
        "request": request_descriptor,
        "num_variables": args.num_variables,
        "num_states_sweep": num_states_sweep,
        "batch_sizes_requested": batch_sizes,
        "num_objects_requested": args.num_objects,
        "max_batch_bytes": args.max_batch_bytes,
        "repeats": args.repeats,
        "payload_descriptors": payload_descriptors,
        "end_to_end": end_to_end,
        "nested_ref_fetch": nested_ref_fetch,
    }
    written = common.write_result("boundary_cost", output_name, result)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
