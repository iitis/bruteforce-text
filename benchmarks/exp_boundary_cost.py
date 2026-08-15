"""E7 - per-unit cost of crossing a node boundary, for extrapolating the controller.

The controller sweep of :mod:`exp_controller_cost` runs on a single machine, so every partial
result it fetches comes from the local object store and every task is scheduled by the local
raylet. That measures the arithmetic and the local scheduling, which are unavoidable, but it
contains no network transfer at all - and the network is precisely what a merge on a large
allocation would have to pay. Its numbers are therefore a lower bound, not an estimate.

This experiment measures the two missing unit costs on real hardware, using the one node
boundary a two-node cluster provides:

``task_roundtrip``
    Submit and await trivial tasks pinned to the local node, then the same pinned to the remote
    node. The difference is what one node boundary adds to dispatching a subproblem.

``object_fetch``
    Create partial results *on* a node and fetch them to the controller, once with the objects
    resident locally and once with them resident on the remote node. Each repetition uses a
    fresh batch, because ``ray.get`` pulls a remote object into the local object store and every
    later fetch of it is a local read -- timing repeated fetches of one batch measures shared
    memory, not the network. Each batch is also fetched a second time as a control: for the
    remote node the cold/warm difference is the transfer itself, and for the local node the two
    coincide. The costs are amortized over a bulk fetch, with the pulls pipelined as they would
    be in a real merge, so they are the right quantity to multiply by ``P``.

Multiplying these per-unit costs by ``P`` bounds the network term of a merge over ``P``
subproblems, under the explicit assumption that the costs stay per-unit - i.e. that neither the
head node nor the fabric saturates. That assumption is exactly what a two-node cluster cannot
test, and it should be stated wherever the extrapolation is used.

Usage (on a cluster started with at least two nodes)::

    python benchmarks/exp_boundary_cost.py --topology 2x1
    python benchmarks/exp_boundary_cost.py --topology 2x4 --num-objects 2048
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

import bench_common as common


def _payload(num_variables: int, num_states: int, seed: int = 0):
    """A stand-in for what one worker returns to the controller."""
    from dimod import SampleSet

    rng = np.random.default_rng(seed)
    samples = rng.choice([0, 1], size=(num_states, num_variables))
    energies = rng.normal(size=num_states)
    return SampleSet.from_samples(samples, "BINARY", energies)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topology", default=None, help="expected cluster label, e.g. 2x1")
    parser.add_argument("--num-tasks", type=int, default=2000)
    parser.add_argument("--num-objects", type=int, default=1024)
    parser.add_argument("--num-variables", type=int, default=60)
    parser.add_argument("--num-states", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    topology = common.require_topology(args.topology)
    if topology["num_nodes"] < 2:
        raise SystemExit(
            f"This experiment needs at least two nodes; the cluster reports "
            f"{topology['label']}. Start a 2xM cluster with scripts/ray_cluster.sh."
        )

    local_node_id = ray.get_runtime_context().get_node_id()
    remote_nodes = [n for n in topology["nodes"] if n["node_id"] != local_node_id]
    if not remote_nodes:
        raise SystemExit("Could not identify a node other than the controller's.")
    remote_node_id = remote_nodes[0]["node_id"]
    print(f"controller on {local_node_id[:12]}..., remote node {remote_node_id[:12]}...")

    def pin(node_id):
        return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

    # One CPU per task on purpose: it bounds worker concurrency the way requesting a GPU does
    # for a real subproblem. With num_cpus=0 Ray starts a worker per in-flight task and the
    # per-task figure measures worker startup rather than dispatch.
    @ray.remote
    def _noop():
        return None

    @ray.remote
    def _make(num_variables, num_states, seed):
        return _payload(num_variables, num_states, seed)

    def timed(fn, repeats):
        fn()  # warm-up: the first call to a node pays worker startup
        return float(np.median([_once(fn) for _ in range(repeats)]))

    def _once(fn):
        start = perf_counter()
        fn()
        return perf_counter() - start

    measurements = {}
    for where, node_id in (("local", local_node_id), ("remote", remote_node_id)):
        strategy = pin(node_id)

        def dispatch():
            ray.get([_noop.options(scheduling_strategy=strategy).remote()
                     for _ in range(args.num_tasks)])

        seconds = timed(dispatch, args.repeats)
        measurements[f"task_roundtrip_{where}_seconds"] = seconds
        measurements[f"task_roundtrip_{where}_per_task_seconds"] = seconds / args.num_tasks
        print(
            f"  {args.num_tasks} trivial tasks on the {where} node: {seconds:7.3f} s "
            f"({1e6 * seconds / args.num_tasks:7.1f} us/task)"
        )

    # Fetching the SAME objects repeatedly would measure almost nothing: ray.get pulls a
    # remote object into the local object store, so every fetch after the first is a local
    # shared-memory read. Each repetition therefore creates a fresh batch on the node under
    # test and fetches it exactly once. The second fetch of that same batch is timed too, as a
    # control: for the remote node the cold/warm difference *is* the transfer, and for the
    # local node the two should coincide.
    def measure_fetch(node_id, repeats):
        strategy = pin(node_id)
        cold, warm = [], []
        for _ in range(repeats + 1):  # first pass discarded as warm-up
            refs = [
                _make.options(scheduling_strategy=strategy).remote(
                    args.num_variables, args.num_states, i
                )
                for i in range(args.num_objects)
            ]
            ray.wait(refs, num_returns=len(refs))  # ready, but not yet pulled to the controller
            start = perf_counter()
            ray.get(refs)
            cold.append(perf_counter() - start)
            start = perf_counter()
            ray.get(refs)  # now resident locally
            warm.append(perf_counter() - start)
            del refs
        return float(np.median(cold[1:])), float(np.median(warm[1:]))

    for where, node_id in (("local", local_node_id), ("remote", remote_node_id)):
        cold_seconds, warm_seconds = measure_fetch(node_id, args.repeats)
        measurements[f"object_fetch_{where}_seconds"] = cold_seconds
        measurements[f"object_fetch_{where}_per_object_seconds"] = cold_seconds / args.num_objects
        measurements[f"object_refetch_{where}_per_object_seconds"] = (
            warm_seconds / args.num_objects
        )
        print(
            f"  fetching {args.num_objects} partial results from the {where} node: "
            f"{cold_seconds:7.3f} s ({1e6 * cold_seconds / args.num_objects:7.1f} us/object); "
            f"re-fetching the same batch: {1e6 * warm_seconds / args.num_objects:7.1f} us/object"
        )

    per_object_local = measurements["object_fetch_local_per_object_seconds"]
    per_object_remote = measurements["object_fetch_remote_per_object_seconds"]
    per_task_local = measurements["task_roundtrip_local_per_task_seconds"]
    per_task_remote = measurements["task_roundtrip_remote_per_task_seconds"]
    measurements["object_fetch_boundary_overhead_per_object_seconds"] = (
        per_object_remote - per_object_local
    )
    measurements["task_roundtrip_boundary_overhead_per_task_seconds"] = (
        per_task_remote - per_task_local
    )

    measurements["object_transfer_remote_per_object_seconds"] = (
        measurements["object_fetch_remote_per_object_seconds"]
        - measurements["object_refetch_remote_per_object_seconds"]
    )

    print("\nper-unit cost of one node boundary:")
    print(f"  dispatching a subproblem : {1e6 * (per_task_remote - per_task_local):+8.1f} us")
    print(f"  collecting a partial result: {1e6 * (per_object_remote - per_object_local):+8.1f} us")

    print("\nextrapolated network term of a direct merge (per-unit cost x P):")
    projection = {}
    for k in (10, 14, 16):
        P = 2**k
        projection[str(P)] = {
            "collect_seconds": P * per_object_remote,
            "dispatch_seconds": P * per_task_remote,
        }
        print(
            f"  P=2^{k:<2} = {P:6d}: collecting {P * per_object_remote:7.1f} s, "
            f"dispatching {P * per_task_remote:7.1f} s"
        )
    print(
        "\nThese assume the per-unit costs hold at scale, i.e. that neither the head node nor\n"
        "the fabric saturates. A two-node cluster cannot test that assumption, so the figures\n"
        "are a linear extrapolation of a measured unit cost, not a measurement."
    )

    payload = {
        "topology": topology,
        "num_tasks": args.num_tasks,
        "num_objects": args.num_objects,
        "num_variables": args.num_variables,
        "num_states": args.num_states,
        "repeats": args.repeats,
        "measurements": measurements,
        "linear_projection": projection,
    }
    print(f"\nwrote {common.write_result('boundary_cost', topology['label'], payload)}")


if __name__ == "__main__":
    main()
