"""Wishart planted-ensemble instances: tunably rugged, with a closed-form optimum.

Hamze, Raymond, Pattison, Biswas and Katzgraber, "Wishart planted ensemble: A tunably
rugged pairwise Ising model with a first-order phase transition", Phys. Rev. E 101,
052102 (2020).

Construction: pick a planted state ``t`` in {-1,+1}^N, draw ``G ~ N(0,1)^(N x M)`` with
``M = round(alpha * N)``, and project every column onto the subspace orthogonal to ``t``:

    W = (I - t t^T / N) G          =>   W^T t = 0.

The couplings are the off-diagonal entries of the Wishart matrix, ``J_ij = (W W^T)_ij / N``
for ``i < j``, with zero field. In the convention ``E(s) = sum_{i<j} J_ij s_i s_j`` shared by
all benchmark instances,

    E(s) = ( |W^T s|^2 - tr(W W^T) ) / (2 N)   >=   -tr(W W^T) / (2 N),

with equality exactly when ``W^T s = 0``, which the planted state satisfies by construction
(so does ``-t``: the field is zero, so the Z2 symmetry is exact and both signs are optimal).
Every instance therefore carries an analytically known ground-state energy,

    E_0 = -tr(W W^T) / (2 N),

which independently checks the brute-force certificate on every single instance. The ratio
``alpha = M/N`` tunes the ruggedness of the landscape through the depth of its first-order
barrier; small ``alpha`` is the hard regime for heuristics, at sizes where exhaustive
certification takes seconds.

Instance files are derived deterministically from ``(seed, size, alpha, replica)`` and are
never rewritten once generated, because the instances behind published results are part of
the record. The random streams are platform-stable, but the matrix products go through BLAS
and may differ in the last ulp between machines, so an existing file is the canonical
artifact: it is re-derived from its seed and compared at value level (tolerance far below
every physically meaningful scale), the analytic ground-state energy is evaluated on the
file's own couplings, and a mismatch beyond floating-point noise is an error rather than a
rewrite.

Usage::

    python benchmarks/gen_wishart.py --size 40 --alphas 0.2 0.3 0.5 --replicas 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import bench_common as common

FAMILY = "wishart"


def alpha_tag(alpha: float) -> str:
    """Filename-safe spelling of alpha: ``0.2 -> "0p2"``."""
    return f"{alpha:g}".replace(".", "p")


def instance_path(
    num_variables: int, alpha: float, replica: int, instances_dir=None
) -> Path:
    directory = (
        Path(instances_dir) if instances_dir is not None else common.INSTANCES_DIR / FAMILY
    )
    return directory / f"N{num_variables}_a{alpha_tag(alpha)}_r{replica}.txt"


def build(
    num_variables: int,
    alpha: float,
    replica: int = 0,
    seed: int = common.DEFAULT_SEED,
):
    """Couplings, planted state and metadata of one deterministic instance.

    The random stream is seeded from ``(seed, size, alpha, replica)`` in the style of
    :func:`bench_common.generate_instance`, so the content of an instance does not depend on
    which other sizes or alphas happen to be generated alongside it.

    :returns: ``(couplings, planted, meta)`` where ``couplings`` is strictly upper
        triangular, ``planted`` is the planted configuration in {-1,+1}^N and ``meta`` holds
        the analytic ground-state energy and the ensemble parameters.
    """
    entropy = [
        seed,
        num_variables,
        replica,
        int(round(1000 * alpha)),
        sum(ord(c) for c in FAMILY),
    ]
    rng = np.random.default_rng(entropy)
    num_vectors = max(1, int(round(alpha * num_variables)))

    planted = rng.choice([-1.0, 1.0], size=num_variables)
    gaussian = rng.standard_normal((num_variables, num_vectors))
    projected = gaussian - np.outer(planted, planted @ gaussian) / num_variables
    wishart = projected @ projected.T
    couplings = np.triu(wishart / num_variables, k=1)

    analytic_gs_energy = -float(np.trace(wishart)) / (2.0 * num_variables)
    planted_energy = float(planted @ couplings @ planted)
    if abs(planted_energy - analytic_gs_energy) > 1e-9 * max(1.0, abs(analytic_gs_energy)):
        raise AssertionError(
            f"planted state does not attain the analytic optimum: "
            f"{planted_energy!r} vs {analytic_gs_energy!r}"
        )

    meta = {
        "family": FAMILY,
        "num_variables": num_variables,
        "alpha": alpha,
        "num_vectors": num_vectors,
        "replica": replica,
        "seed": seed,
        "analytic_gs_energy": analytic_gs_energy,
        "energy_lower_bound": analytic_gs_energy,
        "planted_state": "".join("1" if v > 0 else "0" for v in planted),
    }
    return couplings, planted, meta


def _format_instance(couplings: np.ndarray) -> str:
    """The COO text of an instance.

    Values are written positionally (never in exponent notation) because dimod's COO line
    regex silently skips exponent-notation lines; ``format_float_positional`` with
    ``unique=True`` still round-trips every float64 exactly.
    """
    num_variables = couplings.shape[0]
    lines = []
    for i in range(num_variables):
        lines.append(f"{i} {i} 0.0")
        for j in range(i + 1, num_variables):
            value = np.format_float_positional(couplings[i, j], unique=True, trim="0")
            lines.append(f"{i} {j} {value}")
    return "\n".join(lines) + "\n"


def _parse_instance(path: Path, num_variables: int) -> np.ndarray:
    """Strictly upper-triangular couplings of an instance file."""
    couplings = np.zeros((num_variables, num_variables))
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        i_text, j_text, value_text = line.split()
        i, j, value = int(i_text), int(j_text), float(value_text)
        if i != j:
            couplings[min(i, j), max(i, j)] += value
    return couplings


def generate_instance(
    num_variables: int,
    alpha: float,
    replica: int = 0,
    seed: int = common.DEFAULT_SEED,
    instances_dir=None,
):
    """Write one instance file unless it exists, and return ``(path, meta)``.

    An existing file is never rewritten: it is the canonical published artifact. It is
    instead verified against the deterministic re-derivation at value level - BLAS may
    shift the last ulp of the matrix products between machines, so byte identity across
    hosts is not a valid expectation - and the analytic ground-state energy is evaluated
    on the couplings actually stored in the file, which is what both solvers see. A
    discrepancy beyond floating-point noise (wrong seed, edited file, or a changed NumPy
    random stream) is an error.
    """
    derived, planted, meta = build(num_variables, alpha, replica, seed)
    path = instance_path(num_variables, alpha, replica, instances_dir)
    if path.is_file():
        stored = _parse_instance(path, num_variables)
        if not np.allclose(stored, derived, rtol=1e-9, atol=1e-12):
            worst = float(np.max(np.abs(stored - derived)))
            raise RuntimeError(
                f"existing instance {path} deviates from its deterministic re-derivation "
                f"by up to {worst:.3e}, far beyond floating-point noise; likely causes are "
                "a different --seed, an edited file, or a changed NumPy random stream. "
                "Refusing to proceed with inconsistent inputs."
            )
        planted_energy = float(planted @ stored @ planted)
        if abs(planted_energy - meta["analytic_gs_energy"]) > 1e-9 * max(
            1.0, abs(meta["analytic_gs_energy"])
        ):
            raise RuntimeError(
                f"planted state does not attain the analytic optimum on the stored "
                f"couplings of {path}: {planted_energy!r} vs {meta['analytic_gs_energy']!r}"
            )
        meta = dict(meta, analytic_gs_energy=planted_energy, energy_lower_bound=planted_energy)
        return path, meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_instance(derived))
    return path, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    parser.add_argument("--replicas", type=int, default=5, help="instances per alpha")
    parser.add_argument("--seed", type=int, default=common.DEFAULT_SEED)
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=None,
        help="override the output directory (default: benchmarks/instances/wishart)",
    )
    args = parser.parse_args()

    for alpha in args.alphas:
        energies = []
        for replica in range(args.replicas):
            path, meta = generate_instance(
                args.size, alpha, replica, args.seed, args.instances_dir
            )
            energies.append(meta["analytic_gs_energy"])
            print(f"{path}  E_0 = {meta['analytic_gs_energy']:.12f}")
        print(
            f"alpha={alpha:g}: {args.replicas} instance(s), N={args.size}, "
            f"M={meta['num_vectors']}, analytic E_0 in "
            f"[{min(energies):.4f}, {max(energies):.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
