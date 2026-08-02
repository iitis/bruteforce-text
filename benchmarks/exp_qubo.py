"""E5 - QUBO instances alongside the Ising ones.

Reviewer 3 noted that no QUBO results are reported even though QUBO is a primary target of
the framework. Both samplers convert a SPIN model into its BINARY equivalent, solve it as a
QUBO on the GPU and convert the result back, so the published timings already *are* QUBO
timings. This experiment demonstrates that rather than asserting it: the same coefficient
files are loaded once with ``vartype="SPIN"`` and once with ``vartype="BINARY"``, and the two
runtimes are compared.

The energies of the two runs are not expected to agree - the same coefficients define
different objective functions under the two variable types - but the runtimes are, since the
enumeration is identical and only an O(N^2) host-side transformation differs.

Usage::

    python benchmarks/exp_qubo.py --sizes 40 42 44 46 48 50
"""

from __future__ import annotations

import argparse

import numpy as np

import bench_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[40, 42, 44, 46, 48, 50])
    parser.add_argument("--num-states", type=int, default=1)
    parser.add_argument(
        "--vartypes", nargs="+", default=["SPIN", "BINARY"], choices=["SPIN", "BINARY"]
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="use the distributed sampler on the running Ray cluster instead of one GPU",
    )
    parser.add_argument("--topology", default=None)
    args = parser.parse_args()

    if args.distributed:
        from omnisolver.bruteforce.gpu.distributed import DistributedBruteforceGPUSampler

        topology = common.require_topology(args.topology)
        num_fixed_vars = common.num_fixed_vars_for(topology["total_gpus"])
        sampler = DistributedBruteforceGPUSampler()

        def solve(bqm):
            return sampler.sample(
                bqm,
                num_states=args.num_states,
                num_fixed_vars=num_fixed_vars,
                **common.KERNEL_PARAMS,
            )

    else:
        from omnisolver.bruteforce.gpu import BruteforceGPUSampler

        topology = {"label": "single-gpu", "total_gpus": 1, "num_nodes": 1}
        sampler = BruteforceGPUSampler()

        def solve(bqm):
            return sampler.sample(bqm, num_states=args.num_states, **common.KERNEL_PARAMS)

    measurements = []
    for size in args.sizes:
        instance_file = common.generate_instance(size)
        per_vartype = {}
        for vartype in args.vartypes:
            result = solve(common.load_bqm(instance_file, vartype=vartype))
            per_vartype[vartype] = {
                "solve_time_in_seconds": float(result.info["solve_time_in_seconds"]),
                "energy": float(result.first.energy),
                "state": np.asarray(result.samples()[0, range(size)]),
                "timings": dict(result.info),
            }
            print(
                f"N={size} vartype={vartype:6s} "
                f"t={per_vartype[vartype]['solve_time_in_seconds']:10.3f}s "
                f"energy={per_vartype[vartype]['energy']!r}"
            )

        entry = {"num_variables": size, "by_vartype": per_vartype}
        if {"SPIN", "BINARY"} <= set(per_vartype):
            entry["binary_over_spin_time_ratio"] = (
                per_vartype["BINARY"]["solve_time_in_seconds"]
                / per_vartype["SPIN"]["solve_time_in_seconds"]
            )
            print(f"N={size} BINARY/SPIN runtime ratio = {entry['binary_over_spin_time_ratio']:.3f}")
        measurements.append(entry)

    payload = {
        "sampler_mode": "distributed" if args.distributed else "single-gpu",
        "topology": topology,
        "kernel_params": dict(common.KERNEL_PARAMS, num_states=args.num_states),
        "measurements": measurements,
    }
    sizes = "-".join(str(size) for size in args.sizes)
    name = f"{topology['label']}_N{sizes}"
    print(f"wrote {common.write_result('qubo', name, payload)}")


if __name__ == "__main__":
    main()
