"""Run dSB and simulated annealing against the certified instance-family results.

The source result is treated as immutable evidence: its SHA-256 digest is recorded in the
new result and checked again before output is written.  No GPU or Ray cluster is required.

Examples::

    python benchmarks/exp_multi_solver.py
    python benchmarks/exp_multi_solver.py --seed-bases 420000 430000
    python benchmarks/exp_multi_solver.py --num-reads 32 --num-sweeps 100 \
        --output /tmp/multi_solver_smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

import bench_common as common
import sbm


DEFAULT_SOURCE = common.RESULTS_DIR / "instance_families" / "N44.json"
DEFAULT_NUM_READS = 4096
DEFAULT_NUM_SWEEPS = 3000
DEFAULT_SEED_BASES = (420000,)
REQUIRED_MEASUREMENT_KEYS = {
    "family",
    "replica",
    "num_variables",
    "instance",
    "bf_energy",
    "bf_state_energy",
    "bf_time_seconds",
    "bf_state",
    "sbm_energy",
    "sbm_time_seconds",
    "sbm_state",
}


class SourceError(ValueError):
    """Raised when the BF+dSB source result is incomplete or inconsistent."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def seed_value(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed < 2**31:
        raise argparse.ArgumentTypeError("must be in the range [0, 2**31)")
    return parsed


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(common.BENCHMARKS_DIR))
    except ValueError:
        return str(resolved)


def close_energy(left: float, right: float, atol: float, rtol: float) -> bool:
    return math.isclose(left, right, abs_tol=atol, rel_tol=rtol)


def gap_percent(gap: float, certified_energy: float):
    if certified_energy == 0:
        return None
    return 100.0 * gap / abs(certified_energy)


def load_source(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise SourceError(f"source result does not exist: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceError(f"source result is not valid JSON: {error}") from error

    for key in ("num_variables", "families", "replicas_per_family", "measurements"):
        if key not in payload:
            raise SourceError(f"source result is missing top-level key {key!r}")
    if payload.get("experiment") not in (None, "instance_families"):
        raise SourceError(
            "source result must be an instance_families experiment, got "
            f"{payload.get('experiment')!r}"
        )

    num_variables = payload["num_variables"]
    families = payload["families"]
    replicas = payload["replicas_per_family"]
    rows = payload["measurements"]
    if not isinstance(num_variables, int) or num_variables <= 0:
        raise SourceError("num_variables must be a positive integer")
    if not isinstance(families, list) or not families or len(set(families)) != len(families):
        raise SourceError("families must be a non-empty list without duplicates")
    if not isinstance(replicas, int) or replicas <= 0:
        raise SourceError("replicas_per_family must be a positive integer")
    if not isinstance(rows, list) or len(rows) != len(families) * replicas:
        raise SourceError(
            "measurement count does not match families times replicas_per_family: "
            f"{len(rows) if isinstance(rows, list) else 'not a list'} versus "
            f"{len(families) * replicas}"
        )

    seen = set()
    replicas_by_family = {family: set() for family in families}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SourceError(f"measurement {index} is not an object")
        missing = REQUIRED_MEASUREMENT_KEYS - row.keys()
        if missing:
            raise SourceError(f"measurement {index} is missing {sorted(missing)}")
        family = row["family"]
        replica = row["replica"]
        if family not in replicas_by_family:
            raise SourceError(f"measurement {index} has unknown family {family!r}")
        if row["num_variables"] != num_variables:
            raise SourceError(f"measurement {index} has inconsistent num_variables")
        key = (family, replica)
        if key in seen:
            raise SourceError(f"duplicate measurement for {family}, replica {replica}")
        seen.add(key)
        replicas_by_family[family].add(replica)

    expected_replicas = set(range(replicas))
    for family, observed in replicas_by_family.items():
        if observed != expected_replicas:
            raise SourceError(
                f"family {family!r} has replicas {sorted(observed)}, expected "
                f"{sorted(expected_replicas)}"
            )
    return payload, sha256_bytes(raw)


def spin_state(raw_state, num_variables: int, label: str) -> list[int]:
    if not isinstance(raw_state, list) or len(raw_state) != num_variables:
        raise SourceError(f"{label} must contain {num_variables} spins")
    state = [int(value) for value in raw_state]
    if any(value not in (-1, 1) for value in state):
        raise SourceError(f"{label} contains a value other than -1 or +1")
    return state


def resolve_instance(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = common.BENCHMARKS_DIR / path
    path = path.resolve()
    if not path.is_file():
        raise SourceError(f"instance file does not exist: {path}")
    return path


def source_reference(row: dict, bqm, atol: float, rtol: float) -> dict:
    num_variables = row["num_variables"]
    bf_state = spin_state(row["bf_state"], num_variables, "bf_state")
    dsb_state = spin_state(row["sbm_state"], num_variables, "sbm_state")
    expected_variables = set(range(num_variables))
    if set(bqm.variables) != expected_variables:
        raise SourceError(
            f"instance variables are {sorted(bqm.variables)}, expected 0..{num_variables - 1}"
        )

    bf_state_energy = float(bqm.energy(dict(enumerate(bf_state))))
    certified_energy = float(row["bf_state_energy"])
    if not close_energy(bf_state_energy, certified_energy, atol, rtol):
        raise SourceError(
            f"stored certified energy {certified_energy} disagrees with the instance "
            f"recomputation {bf_state_energy}"
        )

    dsb_state_energy = float(bqm.energy(dict(enumerate(dsb_state))))
    stored_dsb_energy = float(row["sbm_energy"])
    if not close_energy(dsb_state_energy, stored_dsb_energy, atol, rtol):
        raise SourceError(
            f"stored dSB energy {stored_dsb_energy} disagrees with the instance "
            f"recomputation {dsb_state_energy}"
        )
    dsb_gap = dsb_state_energy - certified_energy

    return {
        "bruteforce": {
            "reported_energy": float(row["bf_energy"]),
            "certified_state_energy": certified_energy,
            "wall_time_seconds": float(row["bf_time_seconds"]),
            "state": bf_state,
        },
        "dsb": {
            "reported_energy": stored_dsb_energy,
            "state_energy": dsb_state_energy,
            "gap": dsb_gap,
            "gap_percent": gap_percent(dsb_gap, certified_energy),
            "reached_optimum": close_energy(dsb_state_energy, certified_energy, atol, rtol),
            "hamming_distance": int(np.count_nonzero(np.asarray(dsb_state) != bf_state)),
            "num_optimal_replicas": row.get("sbm_num_optimal_replicas"),
            "wall_time_seconds": float(row["sbm_time_seconds"]),
            "state": dsb_state,
        },
    }


def sample_one_seed(
    sampler,
    bqm,
    certified_energy: float,
    bf_state: list[int],
    seed: int,
    args,
) -> dict:
    started = time.perf_counter()
    sampleset = sampler.sample(
        bqm,
        beta_range=args.beta_range,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        num_sweeps_per_beta=args.num_sweeps_per_beta,
        beta_schedule_type=args.beta_schedule_type,
        seed=seed,
        initial_states_generator="random",
        randomize_order=args.randomize_order,
        proposal_acceptance_criteria=args.proposal_acceptance_criteria,
    )
    wall_time = time.perf_counter() - started
    if not len(sampleset):
        raise RuntimeError(f"simulated annealing returned no samples for seed {seed}")

    best = sampleset.first
    best_state = [int(best.sample[index]) for index in range(len(bf_state))]
    best_state_energy = float(bqm.energy(dict(enumerate(best_state))))
    reported_energy = float(best.energy)
    gap = best_state_energy - certified_energy
    tolerance = args.energy_atol + args.energy_rtol * max(
        abs(best_state_energy), abs(certified_energy)
    )
    if gap < -tolerance:
        raise RuntimeError(
            f"SA energy {best_state_energy} is below certified energy {certified_energy}; "
            "the source result or energy convention is inconsistent"
        )

    energies = np.asarray(sampleset.record.energy, dtype=float)
    occurrences = np.asarray(sampleset.record.num_occurrences, dtype=np.int64)
    optimum_mask = np.isclose(
        energies, certified_energy, atol=args.energy_atol, rtol=args.energy_rtol
    )
    total_reads = int(occurrences.sum())
    optimal_reads = int(occurrences[optimum_mask].sum())
    info = sampleset.info
    timing = {name: int(value) for name, value in info.get("timing", {}).items()}
    return {
        "seed": seed,
        "best_reported_energy": reported_energy,
        "best_state_energy": best_state_energy,
        "energy_recalc_delta": reported_energy - best_state_energy,
        "best_gap": gap,
        "best_gap_percent": gap_percent(gap, certified_energy),
        "reached_optimum": close_energy(
            best_state_energy, certified_energy, args.energy_atol, args.energy_rtol
        ),
        "hamming_distance": int(np.count_nonzero(np.asarray(best_state) != bf_state)),
        "optimal_reads": optimal_reads,
        "total_reads": total_reads,
        "optimal_read_fraction": optimal_reads / total_reads,
        "wall_time_seconds": wall_time,
        "sampler_timing_nanoseconds": timing,
        "effective_beta_range": [float(value) for value in info.get("beta_range", [])],
        "best_state": best_state,
    }


def sample_dsb(bqm, instance_path: Path, certified_energy: float, bf_state: list[int], args) -> dict:
    """Run the shipped dSB implementation on the same host as simulated annealing."""
    instance = sbm.load_instance(instance_path)
    result = sbm.solve(
        instance,
        num_replicas=args.dsb_num_replicas,
        num_steps=args.dsb_num_steps,
        seed=args.dsb_seed,
    )
    state = [int(value) for value in result.state]
    state_energy = float(bqm.energy(dict(enumerate(state))))
    if not close_energy(result.energy, state_energy, args.energy_atol, args.energy_rtol):
        raise RuntimeError(
            f"dSB reported energy {result.energy} disagrees with independent "
            f"recomputation {state_energy} for {instance_path}"
        )
    gap = state_energy - certified_energy
    tolerance = args.energy_atol + args.energy_rtol * max(
        abs(state_energy), abs(certified_energy)
    )
    if gap < -tolerance:
        raise RuntimeError(
            f"dSB energy {state_energy} is below certified energy {certified_energy}; "
            "the source result or energy convention is inconsistent"
        )
    return {
        "seed": args.dsb_seed,
        "reported_energy": float(result.energy),
        "state_energy": state_energy,
        "energy_recalc_delta": float(result.energy) - state_energy,
        "gap": gap,
        "gap_percent": gap_percent(gap, certified_energy),
        "reached_optimum": close_energy(
            state_energy, certified_energy, args.energy_atol, args.energy_rtol
        ),
        "hamming_distance": int(np.count_nonzero(np.asarray(state) != bf_state)),
        "num_optimal_replicas": result.num_optimal_replicas,
        "num_replicas": result.num_replicas,
        "num_steps": result.num_steps,
        "selected_dt": result.dt,
        "wall_time_seconds": result.time_in_seconds,
        "state": state,
    }


def summarize_family(family: str, rows: list[dict]) -> dict:
    bf_rows = [row["reference"]["bruteforce"] for row in rows]
    dsb_rows = [row["dsb"] for row in rows]
    sa_rows = [row["simulated_annealing"] for row in rows]
    seed_runs = [run for row in sa_rows for run in row["seed_runs"]]
    total_reads = sum(run["total_reads"] for run in seed_runs)
    optimal_reads = sum(run["optimal_reads"] for run in seed_runs)
    return {
        "family": family,
        "num_instances": len(rows),
        "bruteforce": {
            "certified_instances": len(rows),
            "median_wall_time_seconds": statistics.median(
                row["wall_time_seconds"] for row in bf_rows
            ),
        },
        "dsb": {
            "instance_optimum_hits": sum(row["reached_optimum"] for row in dsb_rows),
            "instance_total": len(dsb_rows),
            "identical_state_hits": sum(row["hamming_distance"] == 0 for row in dsb_rows),
            "median_best_gap": statistics.median(row["gap"] for row in dsb_rows),
            "maximum_best_gap": max(row["gap"] for row in dsb_rows),
            "median_wall_time_seconds": statistics.median(
                row["wall_time_seconds"] for row in dsb_rows
            ),
        },
        "simulated_annealing": {
            "instance_optimum_hits": sum(row["reached_optimum"] for row in sa_rows),
            "instance_total": len(sa_rows),
            "identical_state_hits": sum(row["hamming_distance"] == 0 for row in sa_rows),
            "seed_run_optimum_hits": sum(run["reached_optimum"] for run in seed_runs),
            "seed_run_total": len(seed_runs),
            "optimal_reads": optimal_reads,
            "total_reads": total_reads,
            "optimal_read_fraction": optimal_reads / total_reads,
            "median_best_gap": statistics.median(row["best_gap"] for row in sa_rows),
            "maximum_best_gap": max(row["best_gap"] for row in sa_rows),
            "median_best_gap_percent": statistics.median(
                row["best_gap_percent"] for row in sa_rows
            ),
            "maximum_best_gap_percent": max(row["best_gap_percent"] for row in sa_rows),
            "median_seed_run_wall_time_seconds": statistics.median(
                run["wall_time_seconds"] for run in seed_runs
            ),
            "total_wall_time_seconds": sum(run["wall_time_seconds"] for run in seed_runs),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"BF+dSB instance-family result (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output JSON (default: results/multi_solver/N<source size>.json)",
    )
    parser.add_argument("--num-reads", type=positive_int, default=DEFAULT_NUM_READS)
    parser.add_argument("--num-sweeps", type=positive_int, default=DEFAULT_NUM_SWEEPS)
    parser.add_argument("--num-sweeps-per-beta", type=positive_int, default=1)
    parser.add_argument(
        "--dsb-num-replicas",
        type=positive_int,
        help="dSB replicas (default: value recorded in the source result)",
    )
    parser.add_argument(
        "--dsb-num-steps",
        type=positive_int,
        help="dSB integration steps (default: value recorded in the source result)",
    )
    parser.add_argument("--dsb-seed", type=seed_value, default=42)
    parser.add_argument(
        "--seed-bases",
        nargs="+",
        type=seed_value,
        default=list(DEFAULT_SEED_BASES),
        help=(
            "base seed(s); each actual seed is BASE + the zero-based source-row index "
            "(default: 420000)"
        ),
    )
    parser.add_argument(
        "--beta-schedule-type", choices=("linear", "geometric"), default="geometric"
    )
    parser.add_argument(
        "--beta-range",
        nargs=2,
        type=float,
        metavar=("BETA_MIN", "BETA_MAX"),
        help="fixed inverse-temperature endpoints (default: deterministic sampler auto-range)",
    )
    parser.add_argument(
        "--randomize-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="randomize spin-update order",
    )
    parser.add_argument(
        "--proposal-acceptance-criteria",
        choices=("Metropolis", "Gibbs"),
        default="Metropolis",
    )
    parser.add_argument("--energy-atol", type=nonnegative_float, default=1e-9)
    parser.add_argument("--energy-rtol", type=nonnegative_float, default=1e-12)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file (never enabled implicitly)",
    )
    return parser


def run(args) -> Path:
    source_path = args.source.resolve()
    source, source_hash = load_source(source_path)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else common.RESULTS_DIR / "multi_solver" / f"N{source['num_variables']}.json"
    )
    if output_path.resolve() == source_path:
        raise ValueError("output path must differ from the immutable source result")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing result {output_path}; pass --overwrite explicitly"
        )
    if len(set(args.seed_bases)) != len(args.seed_bases):
        raise ValueError("--seed-bases must not contain duplicates")
    if max(args.seed_bases) + len(source["measurements"]) - 1 >= 2**31:
        raise ValueError("a derived seed would fall outside the supported range [0, 2**31)")
    if args.beta_range is not None:
        beta_min, beta_max = args.beta_range
        if not (math.isfinite(beta_min) and math.isfinite(beta_max)):
            raise ValueError("--beta-range endpoints must be finite")
        if beta_min <= 0 or beta_max <= beta_min:
            raise ValueError("--beta-range requires 0 < BETA_MIN < BETA_MAX")

    source_sbm_config = source.get("sbm_config")
    if not isinstance(source_sbm_config, dict):
        raise SourceError("source result is missing its sbm_config")
    if args.dsb_num_replicas is None:
        args.dsb_num_replicas = source_sbm_config.get("num_replicas")
    if args.dsb_num_steps is None:
        args.dsb_num_steps = source_sbm_config.get("num_steps")
    if not isinstance(args.dsb_num_replicas, int) or args.dsb_num_replicas <= 0:
        raise SourceError("source sbm_config has no valid num_replicas")
    if not isinstance(args.dsb_num_steps, int) or args.dsb_num_steps <= 0:
        raise SourceError("source sbm_config has no valid num_steps")

    try:
        from dwave.samplers import SimulatedAnnealingSampler
    except ImportError as error:
        raise RuntimeError(
            "dwave-samplers is required; create/update benchmarks/environment.yml"
        ) from error
    try:
        sampler_version = version("dwave-samplers")
    except PackageNotFoundError:
        sampler_version = None

    sampler = SimulatedAnnealingSampler()
    measurements = []
    for measurement_index, row in enumerate(source["measurements"]):
        family = row["family"]
        replica = row["replica"]
        instance_path = resolve_instance(row["instance"])
        instance_hash = sha256_file(instance_path)
        bqm = common.load_bqm(instance_path)
        reference = source_reference(row, bqm, args.energy_atol, args.energy_rtol)
        certified_energy = reference["bruteforce"]["certified_state_energy"]
        bf_state = reference["bruteforce"]["state"]

        dsb_result = sample_dsb(
            bqm, instance_path, certified_energy, bf_state, args
        )
        print(
            f"{family:9s} r{replica} dSB "
            f"gap={dsb_result['gap']:+.3e} "
            f"time={dsb_result['wall_time_seconds']:.3f}s"
        )

        seed_runs = []
        for seed_base in args.seed_bases:
            seed = seed_base + measurement_index
            result = sample_one_seed(
                sampler, bqm, certified_energy, bf_state, seed, args
            )
            result["seed_base"] = seed_base
            result["source_measurement_index"] = measurement_index
            seed_runs.append(result)
            print(
                f"{family:9s} r{replica} seed={seed} "
                f"gap={result['best_gap']:+.3e} "
                f"optimal_reads={result['optimal_reads']}/{result['total_reads']} "
                f"time={result['wall_time_seconds']:.3f}s"
            )

        if sha256_file(instance_path) != instance_hash:
            raise RuntimeError(f"instance changed while it was being sampled: {instance_path}")
        best_run = min(
            seed_runs,
            key=lambda result: (
                result["best_state_energy"],
                result["hamming_distance"],
                result["seed"],
            ),
        )
        sa_result = {
            "best_seed": best_run["seed"],
            "best_reported_energy": best_run["best_reported_energy"],
            "best_state_energy": best_run["best_state_energy"],
            "best_gap": best_run["best_gap"],
            "best_gap_percent": best_run["best_gap_percent"],
            "reached_optimum": any(run["reached_optimum"] for run in seed_runs),
            "hamming_distance": best_run["hamming_distance"],
            "optimum_seed_runs": sum(run["reached_optimum"] for run in seed_runs),
            "seed_run_total": len(seed_runs),
            "optimal_reads": sum(run["optimal_reads"] for run in seed_runs),
            "total_reads": sum(run["total_reads"] for run in seed_runs),
            "best_state": best_run["best_state"],
            "seed_runs": seed_runs,
        }
        sa_result["optimal_read_fraction"] = (
            sa_result["optimal_reads"] / sa_result["total_reads"]
        )
        measurements.append(
            {
                "family": family,
                "replica": replica,
                "source_measurement_index": measurement_index,
                "num_variables": row["num_variables"],
                "instance": display_path(instance_path),
                "instance_sha256": instance_hash,
                "reference": reference,
                "dsb": dsb_result,
                "simulated_annealing": sa_result,
            }
        )

    if sha256_file(source_path) != source_hash:
        raise RuntimeError(f"source result changed during the experiment: {source_path}")

    family_summary = [
        summarize_family(
            family, [row for row in measurements if row["family"] == family]
        )
        for family in source["families"]
    ]
    environment = common.environment_descriptors()
    environment.setdefault("packages", {})["dwave-samplers"] = sampler_version
    payload = {
        "schema_version": 2,
        "experiment": "multi_solver",
        "source": {
            "path": display_path(source_path),
            "sha256": source_hash,
            "experiment": source.get("experiment"),
            "num_variables": source["num_variables"],
            "families": source["families"],
            "replicas_per_family": source["replicas_per_family"],
            "sbm_config": source.get("sbm_config"),
            "kernel_params": source.get("kernel_params"),
            "environment": source.get("environment"),
        },
        "dsb_solver": {
            "name": "discrete_simulated_bifurcation",
            "implementation": "benchmarks.sbm.solve",
            "parameters": {
                "num_replicas": args.dsb_num_replicas,
                "num_steps": args.dsb_num_steps,
                "seed": args.dsb_seed,
                "dt": "automatic selection from sbm.DT_CANDIDATES",
                "dtype": "float64",
            },
        },
        "simulated_annealing_solver": {
            "name": "simulated_annealing",
            "implementation": "dwave.samplers.SimulatedAnnealingSampler",
            "package": "dwave-samplers",
            "package_version": sampler_version,
            "parameters": {
                "num_reads": args.num_reads,
                "num_sweeps": args.num_sweeps,
                "num_sweeps_per_beta": args.num_sweeps_per_beta,
                "seed_bases": args.seed_bases,
                "seed_derivation": "seed_base + zero_based_source_measurement_index",
                "beta_schedule_type": args.beta_schedule_type,
                "requested_beta_range": args.beta_range,
                "initial_states_generator": "random",
                "randomize_order": args.randomize_order,
                "proposal_acceptance_criteria": args.proposal_acceptance_criteria,
            },
        },
        "energy_comparison": {
            "absolute_tolerance": args.energy_atol,
            "relative_tolerance": args.energy_rtol,
            "certified_energy_field": "bf_state_energy",
            "selected_best_states_recomputed_with": "dimod.BinaryQuadraticModel.energy",
            "sa_read_energy_counts_from": "dwave SampleSet.record.energy",
        },
        "provenance": {
            "driver": display_path(Path(__file__)),
            "driver_sha256": sha256_file(Path(__file__)),
            "dsb_implementation": display_path(Path(sbm.__file__)),
            "dsb_implementation_sha256": sha256_file(Path(sbm.__file__)),
            "shared_helpers": display_path(Path(common.__file__)),
            "shared_helpers_sha256": sha256_file(Path(common.__file__)),
        },
        "measurements": measurements,
        "family_summary": family_summary,
        "environment": environment,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    with output_path.open(mode) as output:
        json.dump(payload, output, indent=4, cls=common.NumpyEncoder)
        output.write("\n")
    print(f"wrote {output_path}")
    for summary in family_summary:
        dsb = summary["dsb"]
        sa = summary["simulated_annealing"]
        print(
            f"{summary['family']:9s}: dSB {dsb['instance_optimum_hits']}/"
            f"{dsb['instance_total']}, SA {sa['instance_optimum_hits']}/"
            f"{sa['instance_total']} certified optima"
        )
    return output_path


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, RuntimeError, SourceError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
