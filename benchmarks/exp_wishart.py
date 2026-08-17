"""E9 - a planted family on which the heuristic fails and certification is decisive.

The four families of E6 are all solved by the shipped dSB heuristic under the article's
budgets, which demonstrates agreement but not necessity. The Wishart planted ensemble of
:mod:`gen_wishart` supplies the complement: at small alpha its first-order landscape defeats
the heuristic at sizes the brute-force sampler certifies in seconds, and its closed-form
ground-state energy E_0 checks every certificate independently of both solvers.

For every (alpha, replica) the driver derives the deterministic instance, certifies it with
the single-GPU brute-force sampler (float32 fast path, float64 re-evaluation), verifies the
certificate against the analytic E_0 - which is also the unconstrained lower bound, so a
certificate below it halts the analysis - runs the shipped dSB heuristic of :mod:`sbm` under
the same budgets as E6/E8, and records whether the heuristic reached the certified optimum.
Scoring is on energy; Hamming distances are informational, because the zero field makes the
Z2-mirrored optimum exactly degenerate.

The analytic optimum also makes a GPU-free re-check possible: with ``--skip-bruteforce`` the
heuristic is scored against E_0 instead of against a fresh certificate.

Usage::

    python benchmarks/exp_wishart.py --size 40 --alphas 0.2 0.3 0.5 --replicas 5
    python benchmarks/exp_wishart.py --skip-bruteforce   # CPU-only, scores against E_0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import bench_common as common
import gen_wishart
import sbm


class CertificateViolation(RuntimeError):
    """An energy below the analytic lower bound: impossible, so the analysis must halt."""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--size",
        type=int,
        default=40,
        help="problem size; the brute-force cost doubles with every added variable",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.5],
        help="Wishart alpha = M/N values; small alpha is the hard regime",
    )
    parser.add_argument("--replicas", type=int, default=5, help="instances per alpha")
    parser.add_argument("--num-replicas", type=int, default=sbm.DEFAULT_NUM_REPLICAS)
    parser.add_argument("--num-steps", type=int, default=sbm.DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=42, help="dSB random seed")
    parser.add_argument(
        "--skip-bruteforce",
        action="store_true",
        help="only run the heuristic and score it against the analytic E_0 (no GPU needed)",
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=None,
        help="override the instance directory (default: benchmarks/instances/wishart)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output JSON (default: a timestamped file under results/wishart)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an explicitly selected output file (never enabled implicitly)",
    )
    args = parser.parse_args()

    sampler = None
    if not args.skip_bruteforce:
        from omnisolver.bruteforce.gpu import BruteforceGPUSampler

        sampler = BruteforceGPUSampler()

    measurements = []
    for alpha in args.alphas:
        for replica in range(args.replicas):
            instance_file, meta = gen_wishart.generate_instance(
                args.size, alpha, replica, instances_dir=args.instances_dir
            )
            reference = sbm.load_instance(instance_file)
            analytic = meta["analytic_gs_energy"]
            planted = np.array(
                [1.0 if bit == "1" else -1.0 for bit in meta["planted_state"]]
            )

            try:
                instance_label = str(instance_file.relative_to(common.BENCHMARKS_DIR))
            except ValueError:
                instance_label = str(instance_file)
            entry = {
                "family": "wishart",
                "alpha": alpha,
                "replica": replica,
                "num_variables": args.size,
                "num_vectors": meta["num_vectors"],
                "instance": instance_label,
                "analytic_gs_energy": analytic,
            }

            if sampler is not None:
                bqm = common.load_bqm(instance_file)
                if bqm.num_interactions != args.size * (args.size - 1) // 2:
                    raise RuntimeError(
                        f"{instance_file} loaded with {bqm.num_interactions} couplings, "
                        f"expected {args.size * (args.size - 1) // 2}: the sampler would "
                        "solve a different Hamiltonian than the one scored"
                    )
                bf = sampler.sample(bqm, num_states=1, **common.KERNEL_PARAMS)
                bf_state = np.asarray(bf.samples()[0, range(args.size)])
                bf_state_energy = reference.energy(bf_state)
                if bf_state_energy < analytic - 1e-9 * max(1.0, abs(analytic)):
                    raise CertificateViolation(
                        f"certified energy {bf_state_energy!r} lies below the analytic "
                        f"lower bound {analytic!r} on {instance_file}"
                    )
                certificate_deviation = bf_state_energy - analytic
                entry.update(
                    {
                        "bf_energy": float(bf.first.energy),
                        "bf_state_energy": bf_state_energy,
                        "bf_energy_recalc_delta": float(bf.first.energy) - bf_state_energy,
                        "bf_time_seconds": float(bf.info["solve_time_in_seconds"]),
                        "bf_state": bf_state,
                        "certificate_deviation": certificate_deviation,
                        "certificate_matches_analytic": bool(
                            abs(certificate_deviation) <= 1e-9 * max(1.0, abs(analytic))
                        ),
                        "bf_hamming_to_planted": int(
                            min(
                                np.count_nonzero(bf_state != planted),
                                np.count_nonzero(bf_state != -planted),
                            )
                        ),
                    }
                )
                certified_energy = bf_state_energy
            else:
                certified_energy = analytic

            heuristic = sbm.solve(
                reference,
                num_replicas=args.num_replicas,
                num_steps=args.num_steps,
                seed=args.seed,
            )
            gap = heuristic.energy - certified_energy
            if gap < -1e-6:
                raise CertificateViolation(
                    f"heuristic energy {heuristic.energy!r} lies below the certified "
                    f"optimum {certified_energy!r} on {instance_file}"
                )
            entry.update(
                {
                    "sbm_energy": heuristic.energy,
                    "sbm_time_seconds": heuristic.time_in_seconds,
                    "sbm_num_optimal_replicas": heuristic.num_optimal_replicas,
                    "sbm_selected_dt": heuristic.dt,
                    "sbm_state": heuristic.state,
                    "gap": gap,
                    "gap_percent": (
                        100.0 * gap / abs(certified_energy)
                        if certified_energy
                        else float("nan")
                    ),
                    "sbm_hamming_to_planted": int(
                        min(
                            np.count_nonzero(heuristic.state != planted),
                            np.count_nonzero(heuristic.state != -planted),
                        )
                    ),
                    "sbm_reached_optimum": bool(gap <= 1e-9),
                }
            )
            if sampler is not None:
                entry["hamming_distance"] = int(
                    np.count_nonzero(heuristic.state != entry["bf_state"])
                )

            certificate_note = (
                f"cert_dev={entry['certificate_deviation']:+.1e}  "
                if sampler is not None
                else "vs analytic E_0  "
            )
            print(
                f"wishart a={alpha:<4g} r{replica} N={args.size}  "
                f"E_0={certified_energy:.8f}  sbm={heuristic.energy:.8f}  "
                f"gap={gap:+.2e}  {certificate_note}"
                f"{'reached' if entry['sbm_reached_optimum'] else 'MISSED'}"
            )
            measurements.append(entry)

    payload = {
        "experiment": "wishart",
        "num_variables": args.size,
        "alphas": args.alphas,
        "replicas_per_alpha": args.replicas,
        "sbm_config": {
            "num_replicas": args.num_replicas,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "dt": "automatic selection from sbm.DT_CANDIDATES",
            "dtype": "float64",
        },
        "kernel_params": dict(common.KERNEL_PARAMS, num_states=1),
        "measurements": measurements,
        "environment": common.environment_descriptors(),
    }
    if args.output is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        path = common.RESULTS_DIR / "wishart" / f"N{args.size}_{run_id}.json"
    else:
        path = args.output.resolve()
    if path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing result {path}; pass --overwrite explicitly"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    with path.open(mode) as output:
        json.dump(payload, output, indent=4, cls=common.NumpyEncoder)
        output.write("\n")

    print()
    for alpha in args.alphas:
        rows = [m for m in measurements if m["alpha"] == alpha]
        reached = sum(m["sbm_reached_optimum"] for m in rows)
        worst = max(m["gap_percent"] for m in rows)
        line = (
            f"alpha {alpha:<4g}: heuristic reached the certified optimum in "
            f"{reached}/{len(rows)} instances, worst gap {worst:.6f}%"
        )
        if not args.skip_bruteforce:
            deviation = max(abs(m["certificate_deviation"]) for m in rows)
            line += f", max |certificate - E_0| = {deviation:.2e}"
        print(line)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
