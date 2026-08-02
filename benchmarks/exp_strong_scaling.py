"""E1 - strong scaling: fixed problem size, varying GPU count and node layout.

Answers the reviewers' request for a scaling study that separates the number of GPUs from
the way they are distributed over nodes. Because Ray cannot rearrange a running cluster, the
topology is set by how ``ray start`` was invoked on each node; this script detects it,
refuses to run if it does not match the requested label, and records it alongside the timing.

The intended series is::

    1x1   (the single-GPU sampler, for the reference point)   --single-gpu
    1x1   (the distributed sampler with k=0, i.e. pure Ray overhead)
    1x2   2 GPUs on one node
    2x1   2 GPUs on two nodes      <- same GPU count, one node boundary crossed
    1x4   4 GPUs on one node
    2x2   4 GPUs on two nodes      <- same GPU count, one node boundary crossed
    2x4   8 GPUs on two nodes      (the configuration used for Fig. 1)

Comparing 1x2 against 2x1 and 1x4 against 2x2 isolates the cost of crossing the node
boundary at constant compute, which is the closest available proxy for the multi-node
behaviour the reviewers asked about.

Usage (on each topology, after starting Ray accordingly)::

    python benchmarks/exp_strong_scaling.py --size 50 --topology 1x2
    python benchmarks/exp_strong_scaling.py --size 50 --single-gpu
"""

from __future__ import annotations

import argparse

import numpy as np

import bench_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=50, help="problem size N (default: 50)")
    parser.add_argument(
        "--topology",
        default=None,
        help="expected cluster label, e.g. 1x2 or 2x2; checked against the running cluster",
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="use BruteforceGPUSampler instead of the distributed one (no Ray involved)",
    )
    parser.add_argument(
        "--num-fixed-vars",
        type=int,
        default=None,
        help="override k; by default one subproblem is placed on each GPU",
    )
    parser.add_argument("--num-states", type=int, default=1)
    args = parser.parse_args()

    instance = common.generate_instance(args.size)
    bqm = common.load_bqm(instance)

    if args.single_gpu:
        from omnisolver.bruteforce.gpu import BruteforceGPUSampler

        label, topology = "single-gpu", {"label": "single-gpu", "total_gpus": 1, "num_nodes": 1}
        result = BruteforceGPUSampler().sample(
            bqm, num_states=args.num_states, **common.KERNEL_PARAMS
        )
    else:
        from omnisolver.bruteforce.gpu.distributed import DistributedBruteforceGPUSampler

        topology = common.require_topology(args.topology)
        label = topology["label"]
        num_fixed_vars = (
            args.num_fixed_vars
            if args.num_fixed_vars is not None
            else common.num_fixed_vars_for(topology["total_gpus"])
        )
        result = DistributedBruteforceGPUSampler().sample(
            bqm,
            num_states=args.num_states,
            num_fixed_vars=num_fixed_vars,
            **common.KERNEL_PARAMS,
        )
        topology = dict(topology, num_fixed_vars=num_fixed_vars)

    state = np.asarray(result.samples()[0, range(args.size)])
    payload = {
        "num_variables": args.size,
        "instance": str(instance.relative_to(common.BENCHMARKS_DIR)),
        "sampler_mode": "single-gpu" if args.single_gpu else "distributed",
        "topology": topology,
        "kernel_params": dict(common.KERNEL_PARAMS, num_states=args.num_states),
        "energy": float(result.first.energy),
        "state": state,
        "timings": dict(result.info),
    }
    path = common.write_result("strong_scaling", f"N{args.size}_{label}", payload)

    print(f"N={args.size} topology={label} energy={result.first.energy!r}")
    for key, value in sorted(result.info.items()):
        print(f"  {key} = {value}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
