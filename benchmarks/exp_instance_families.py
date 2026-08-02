"""E6 - certified verification of the heuristic across several instance families.

Reviewer 1 asked for a broader comparison against other solvers and across instance families.
The runtime of an exhaustive search does not depend on the family - all 2 ** N configurations
are enumerated regardless of the coefficients - so a runtime sweep over families would only
reproduce the existing curve. What the family *does* change is the difficulty seen by the
heuristic whose answers the plugin is meant to certify, so this experiment sweeps families and
asks, for each instance, whether the heuristic recovers the certified optimum.

For every family and replica it runs the brute-force solver, recomputes the energy of the
returned configuration in float64, runs the CPU simulated bifurcation solver of :mod:`sbm` on
the same instance, and records the energy gap and the Hamming distance between the two
answers.

Usage::

    python benchmarks/exp_instance_families.py --size 44 --replicas 5
    python benchmarks/exp_instance_families.py --size 48 --replicas 5 --families uniform bimodal
"""

from __future__ import annotations

import argparse

import numpy as np

import bench_common as common
import sbm


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--size",
        type=int,
        default=44,
        help="problem size; the brute-force cost doubles with every added variable",
    )
    parser.add_argument("--replicas", type=int, default=5, help="instances per family")
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(common.INSTANCE_FAMILIES),
        choices=list(common.INSTANCE_FAMILIES),
    )
    parser.add_argument("--num-replicas", type=int, default=sbm.DEFAULT_NUM_REPLICAS)
    parser.add_argument("--num-steps", type=int, default=sbm.DEFAULT_NUM_STEPS)
    parser.add_argument(
        "--skip-bruteforce",
        action="store_true",
        help="only run the heuristic, e.g. to re-analyse an earlier sweep without a GPU",
    )
    args = parser.parse_args()

    sampler = None
    if not args.skip_bruteforce:
        from omnisolver.bruteforce.gpu import BruteforceGPUSampler

        sampler = BruteforceGPUSampler()

    measurements = []
    for family in args.families:
        for replica in range(args.replicas):
            instance_file = common.generate_instance(
                args.size, family=family, replica=replica
            )
            reference = sbm.load_instance(instance_file)

            entry = {
                "family": family,
                "replica": replica,
                "num_variables": args.size,
                "instance": str(instance_file.relative_to(common.BENCHMARKS_DIR)),
            }

            if sampler is not None:
                bf = sampler.sample(
                    common.load_bqm(instance_file), num_states=1, **common.KERNEL_PARAMS
                )
                bf_state = np.asarray(bf.samples()[0, range(args.size)])
                bf_state_energy = reference.energy(bf_state)
                entry.update(
                    {
                        "bf_energy": float(bf.first.energy),
                        "bf_state_energy": bf_state_energy,
                        "bf_energy_recalc_delta": float(bf.first.energy) - bf_state_energy,
                        "bf_time_seconds": float(bf.info["solve_time_in_seconds"]),
                        "bf_state": bf_state,
                    }
                )

            heuristic = sbm.solve(
                reference, num_replicas=args.num_replicas, num_steps=args.num_steps
            )
            entry.update(
                {
                    "sbm_energy": heuristic.energy,
                    "sbm_time_seconds": heuristic.time_in_seconds,
                    "sbm_num_optimal_replicas": heuristic.num_optimal_replicas,
                    "sbm_state": heuristic.state,
                }
            )

            if sampler is not None:
                entry["hamming_distance"] = int(
                    np.count_nonzero(heuristic.state != entry["bf_state"])
                )
                entry["gap"] = heuristic.energy - entry["bf_state_energy"]
                entry["gap_percent"] = (
                    100.0 * entry["gap"] / abs(entry["bf_state_energy"])
                    if entry["bf_state_energy"]
                    else float("nan")
                )
                entry["sbm_reached_optimum"] = bool(
                    entry["hamming_distance"] == 0
                    or entry["gap"] <= 1e-9
                )
                print(
                    f"{family:9s} r{replica} N={args.size}  bf={entry['bf_state_energy']:.8f}  "
                    f"sbm={heuristic.energy:.8f}  gap={entry['gap']:+.2e}  "
                    f"hamming={entry['hamming_distance']:3d}  "
                    f"{'reached' if entry['sbm_reached_optimum'] else 'MISSED'}"
                )
            else:
                print(f"{family:9s} r{replica} N={args.size}  sbm={heuristic.energy:.8f}")

            measurements.append(entry)

    payload = {
        "num_variables": args.size,
        "families": args.families,
        "replicas_per_family": args.replicas,
        "sbm_config": {"num_replicas": args.num_replicas, "num_steps": args.num_steps},
        "kernel_params": dict(common.KERNEL_PARAMS, num_states=1),
        "measurements": measurements,
    }
    path = common.write_result("instance_families", f"N{args.size}", payload)

    if sampler is not None:
        print()
        for family in args.families:
            rows = [m for m in measurements if m["family"] == family]
            reached = sum(m["sbm_reached_optimum"] for m in rows)
            worst = max(m["gap_percent"] for m in rows)
            print(
                f"{family:9s}: heuristic reached the certified optimum in {reached}/{len(rows)} "
                f"instances, worst gap {worst:.6f}%"
            )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
