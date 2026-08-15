"""Cross-check certified brute-force optima against the CPU simulated bifurcation solver.

For every brute-force result stored under ``results/<mode>/``, this script

1. recomputes the energy of the returned configuration from scratch in ``float64``, which
   quantifies the roundoff of the single-precision GPU accumulator;
2. runs the discrete simulated bifurcation heuristic of :mod:`sbm` on the same instance;
3. reports whether the heuristic reached the certified optimum, and at what Hamming distance.

The results are written to the ``bf_sbm_verification_*`` tables, one summary and one states
file per sampler mode. Passing ``--check`` turns the script into a regression gate: it exits
non-zero if the heuristic failed to reach the certified optimum on any instance, which is the
property the verification exists to establish.

Usage::

    python benchmarks/verify_sbm.py                     # all modes, write the tables
    python benchmarks/verify_sbm.py --check             # ... and fail if any optimum was missed
    python benchmarks/verify_sbm.py --mode distributed  # one mode only
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import bench_common as common
import sbm

SUMMARY_COLUMNS = (
    "size",
    "bf_energy",
    "bf_time_seconds",
    "bf_state_energy",
    "bf_energy_recalc_delta",
    "sbm_energy",
    "sbm_vs_bf_recalc_delta",
    "sbm_time_seconds",
    "sbm_num_optimal_replicas",
    "gap_percent",
    "hamming_distance",
    "sbm_reached_bf",
)


def verify_mode(mode: str, num_replicas: int, num_steps: int, seed: int) -> list:
    """Verify every stored brute-force result of one sampler mode."""
    result_dir = common.RESULTS_DIR / mode
    if not result_dir.is_dir():
        raise FileNotFoundError(f"No results directory for mode {mode!r}: {result_dir}")

    verified = []
    for path in sorted(result_dir.glob("*.json"), key=lambda p: int(p.stem)):
        with open(path) as fd:
            bf = json.load(fd)
        size = int(bf["num_variables"])
        instance = sbm.load_instance(common.instance_path(size))

        bf_state = np.asarray(bf["state"], dtype=np.int8)
        bf_state_energy = instance.energy(bf_state)
        bf_energy = float(bf["energy"])

        result = sbm.solve(instance, num_replicas=num_replicas, num_steps=num_steps, seed=seed)
        hamming = int(np.count_nonzero(result.state != bf_state))

        verified.append(
            {
                "size": size,
                "bf_energy": bf_energy,
                "bf_time_seconds": float(bf.get("solve_time_in_seconds", float("nan"))),
                "bf_state_energy": bf_state_energy,
                "bf_energy_recalc_delta": bf_energy - bf_state_energy,
                "sbm_energy": result.energy,
                "sbm_vs_bf_recalc_delta": result.energy - bf_state_energy,
                "sbm_time_seconds": result.time_in_seconds,
                "sbm_num_optimal_replicas": result.num_optimal_replicas,
                "gap_percent": (
                    100.0 * (result.energy - bf_state_energy) / abs(bf_state_energy)
                    if bf_state_energy
                    else float("nan")
                ),
                "hamming_distance": hamming,
                # The heuristic reached the certified optimum iff it returned a configuration
                # of the same energy; comparing float32-rounded reported values would be
                # dominated by roundoff rather than by solution quality.
                "sbm_reached_bf": bool(
                    hamming == 0 or result.energy <= bf_state_energy + 1e-9
                ),
                "sbm_state": result.state,
                "bf_state": bf_state,
                "sbm_dt": result.dt,
            }
        )
    return verified


def write_tables(mode: str, rows: list) -> tuple:
    summary_path = common.BENCHMARKS_DIR / f"bf_sbm_verification_{mode}_summary.csv"
    states_path = common.BENCHMARKS_DIR / f"bf_sbm_verification_{mode}_states.csv"

    with open(summary_path, "w", newline="") as fd:
        writer = csv.writer(fd)
        writer.writerow(SUMMARY_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row["size"],
                    "%.17g" % row["bf_energy"],
                    "%.9f" % row["bf_time_seconds"],
                    "%.17g" % row["bf_state_energy"],
                    "%.17g" % row["bf_energy_recalc_delta"],
                    "%.17g" % row["sbm_energy"],
                    "%.17g" % row["sbm_vs_bf_recalc_delta"],
                    "%.9f" % row["sbm_time_seconds"],
                    row["sbm_num_optimal_replicas"],
                    "%.9f" % row["gap_percent"],
                    row["hamming_distance"],
                    row["sbm_reached_bf"],
                ]
            )

    with open(states_path, "w", newline="") as fd:
        fd.write("size,bf_state,sbm_state\n")
        for row in rows:
            bf = " ".join(str(int(v)) for v in row["bf_state"])
            sb = " ".join(str(int(v)) for v in row["sbm_state"])
            fd.write(f'{row["size"]},"[{bf}]","[{sb}]"\n')

    return summary_path, states_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        action="append",
        choices=["single-gpu", "distributed"],
        help="sampler mode to verify (default: both)",
    )
    parser.add_argument("--num-replicas", type=int, default=sbm.DEFAULT_NUM_REPLICAS)
    parser.add_argument("--num-steps", type=int, default=sbm.DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the heuristic missed the certified optimum on any instance",
    )
    args = parser.parse_args()

    modes = args.mode or ["single-gpu", "distributed"]
    exit_code = 0
    for mode in modes:
        rows = verify_mode(mode, args.num_replicas, args.num_steps, args.seed)
        summary_path, states_path = write_tables(mode, rows)
        reached = sum(row["sbm_reached_bf"] for row in rows)
        worst_bf_roundoff = max(abs(row["bf_energy_recalc_delta"]) for row in rows)
        print(
            f"[{mode}] {len(rows)} instances verified, heuristic reached the certified optimum "
            f"in {reached}/{len(rows)}; largest |reported - recomputed| brute-force energy "
            f"deviation {worst_bf_roundoff:.2e}"
        )
        print(f"[{mode}] wrote {Path(summary_path).name} and {Path(states_path).name}")
        if args.check and reached != len(rows):
            missed = [row["size"] for row in rows if not row["sbm_reached_bf"]]
            print(f"[{mode}] FAILED: optimum not reached at N = {missed}")
            exit_code = 1

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
