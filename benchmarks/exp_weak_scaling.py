"""E2 - weak scaling: constant work per GPU, growing problem and GPU count.

Reviewer 2 asked for weak scaling and gave the right construction: compare N = 50..53 at 1,
2, 4 and 8 GPUs. The point is that with a fixed ``num_fixed_vars`` the number of subproblems
does not follow the device count, so running the same instance on fewer GPUs only makes Ray
queue the subproblems. Weak scaling instead requires k to grow together with the GPU count,
so that the work per GPU, 2 ** (N - k), stays constant:

===========  ===  ===  ===================
Topology     P    k    N = base + k
===========  ===  ===  ===================
1x1          1    0    50
1x2 / 2x1    2    1    51
1x4 / 2x2    4    2    52
2x4          8    3    53
===========  ===  ===  ===================

Ideal weak scaling would keep the wall-clock time constant across the rows; whatever growth
remains is the price of dispatch, of the merge and, when comparing 2xM against 1xM, of
crossing the node boundary.

Usage (on each topology, after starting Ray accordingly)::

    python benchmarks/exp_weak_scaling.py --topology 1x2
"""

from __future__ import annotations

import argparse

import numpy as np

import bench_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-size",
        type=int,
        default=50,
        help="work per GPU, expressed as the problem size solved by a single GPU (default: 50)",
    )
    parser.add_argument("--topology", default=None, help="expected cluster label, e.g. 1x2")
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="reference point: the single-GPU sampler on the base size, without Ray",
    )
    parser.add_argument("--num-states", type=int, default=1)
    args = parser.parse_args()

    if args.single_gpu:
        from omnisolver.bruteforce.gpu import BruteforceGPUSampler

        num_fixed_vars, size, label = 0, args.base_size, "single-gpu"
        topology = {"label": label, "total_gpus": 1, "num_nodes": 1}
        instance = common.generate_instance(size)
        result = BruteforceGPUSampler().sample(
            common.load_bqm(instance), num_states=args.num_states, **common.KERNEL_PARAMS
        )
    else:
        from omnisolver.bruteforce.gpu.distributed import DistributedBruteforceGPUSampler

        topology = common.require_topology(args.topology)
        label = topology["label"]
        num_fixed_vars = common.num_fixed_vars_for(topology["total_gpus"])
        size = args.base_size + num_fixed_vars
        instance = common.generate_instance(size)
        topology = dict(topology, num_fixed_vars=num_fixed_vars)
        result = DistributedBruteforceGPUSampler().sample(
            common.load_bqm(instance),
            num_states=args.num_states,
            num_fixed_vars=num_fixed_vars,
            **common.KERNEL_PARAMS,
        )

    payload = {
        "num_variables": size,
        "base_size": args.base_size,
        "work_per_gpu_log2": size - num_fixed_vars,
        "instance": str(instance.relative_to(common.BENCHMARKS_DIR)),
        "sampler_mode": "single-gpu" if args.single_gpu else "distributed",
        "topology": topology,
        "kernel_params": dict(common.KERNEL_PARAMS, num_states=args.num_states),
        "energy": float(result.first.energy),
        "state": np.asarray(result.samples()[0, range(size)]),
        "timings": dict(result.info),
    }
    path = common.write_result("weak_scaling", f"base{args.base_size}_{label}", payload)

    print(
        f"weak scaling point: N={size}, k={num_fixed_vars}, work per GPU = 2^{size - num_fixed_vars}, "
        f"topology={label}"
    )
    for key, value in sorted(result.info.items()):
        print(f"  {key} = {value}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
