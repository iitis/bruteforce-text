"""E4 - single precision against double precision on the ground-state path.

Reviewer 2 asked for the speedup that the ``float32`` fast path buys over the default double
precision, so that users tracking several low-energy states can estimate the slowdown they
should expect. Two things make this more than a timing ratio:

* the stabilization (compensated updates, periodic exact re-anchoring) is enabled only for
  ``float32`` and only for kernels seeing at least 40 variables, so ``float64`` is a different
  numerical path rather than a slower one;
* the reported energy of a ``float32`` run carries incremental-update roundoff, which is why
  every run below is re-checked against a from-scratch ``float64`` recomputation.

Usage::

    python benchmarks/exp_precision.py --sizes 40 42 44 46
"""

from __future__ import annotations

import argparse

import numpy as np

import bench_common as common
import sbm


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[40, 42, 44, 46])
    parser.add_argument("--num-states", type=int, default=1)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float", "double"],
        choices=["float", "double"],
        help="precisions to measure; 'float' is single precision",
    )
    args = parser.parse_args()

    from omnisolver.bruteforce.gpu import BruteforceGPUSampler

    sampler = BruteforceGPUSampler()
    measurements = []
    for size in args.sizes:
        instance_file = common.generate_instance(size)
        bqm = common.load_bqm(instance_file)
        reference = sbm.load_instance(instance_file)

        per_dtype = {}
        for dtype in args.dtypes:
            result = sampler.sample(
                bqm, num_states=args.num_states, dtype=dtype, **common.KERNEL_PARAMS
            )
            state = np.asarray(result.samples()[0, range(size)])
            recomputed = reference.energy(state)
            per_dtype[dtype] = {
                "solve_time_in_seconds": float(result.info["solve_time_in_seconds"]),
                "reported_energy": float(result.first.energy),
                "recomputed_energy": recomputed,
                "roundoff": float(result.first.energy) - recomputed,
                "state": state,
                # Stabilization is enabled for single precision from 40 variables upwards.
                "stabilization_active": dtype == "float" and size >= 40,
            }
            print(
                f"N={size} dtype={dtype:6s} t={per_dtype[dtype]['solve_time_in_seconds']:10.3f}s "
                f"|reported-recomputed|={abs(per_dtype[dtype]['roundoff']):.2e}"
            )

        entry = {"num_variables": size, "by_dtype": per_dtype}
        if {"float", "double"} <= set(per_dtype):
            slowdown = (
                per_dtype["double"]["solve_time_in_seconds"]
                / per_dtype["float"]["solve_time_in_seconds"]
            )
            entry["double_over_single_slowdown"] = slowdown
            entry["same_configuration"] = bool(
                np.array_equal(per_dtype["float"]["state"], per_dtype["double"]["state"])
            )
            print(
                f"N={size} double/single slowdown = {slowdown:.2f}x, "
                f"same configuration: {entry['same_configuration']}"
            )
        measurements.append(entry)

    payload = {
        "kernel_params": dict(common.KERNEL_PARAMS, num_states=args.num_states),
        "measurements": measurements,
    }
    sizes = "-".join(str(size) for size in args.sizes)
    print(f"wrote {common.write_result('precision', f'N{sizes}', payload)}")


if __name__ == "__main__":
    main()
