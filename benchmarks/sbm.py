"""CPU implementation of the discrete Simulated Bifurcation Machine (dSB).

This module provides a dependency-light reference implementation of the heuristic used to
cross-check the certified ground states produced by ``omnisolver-bruteforce``. It exists so
that the verification reported in the manuscript can be reproduced without access to a GPU
or to a third-party solver: everything here is NumPy.

Conventions
-----------
Instances are read in the COOrdinate format written by :mod:`bf.py`, with ``i j value``
triples, and are interpreted exactly as ``dimod`` interprets them for ``vartype="SPIN"``:
a diagonal entry ``i i value`` is the linear bias :math:`h_i`, an off-diagonal entry is the
coupling :math:`J_{ij}`, and the energy of a spin configuration is

.. math::

    E(s) = \\sum_i h_i s_i + \\sum_{i<j} J_{ij} s_i s_j
         = h \\cdot s + \\tfrac{1}{2}\\, s^{T} W s ,

where :math:`W` is the symmetric matrix with zero diagonal and :math:`W_{ij} = J_{ij}`.

Dynamics
--------
The simulated bifurcation Hamiltonian is descended in the form given by Goto et al.,
adapted to the sign convention above. With the pump ramped linearly from zero to
:math:`a_0`,

.. math::

    \\dot{y}_i &= -\\bigl(a_0 - a(t)\\bigr) x_i - c_0 \\bigl( (W \\operatorname{sign} x)_i
                  + h_i \\bigr) \\\\
    \\dot{x}_i &= a_0 y_i ,

with inelastic walls at :math:`|x_i| = 1`: a replica reaching a wall is clipped and has its
momentum zeroed. Using :math:`\\operatorname{sign}(x)` rather than :math:`x` in the coupling
term is what makes this the *discrete* variant (dSB), which is the variant used for the
verification tables in this repository.

All replicas are integrated simultaneously, so the cost per step is one dense
``(N, N) @ (N, num_replicas)`` product; the total cost is
:math:`O(N^2 \\cdot \\mathrm{num\\_replicas} \\cdot \\mathrm{num\\_steps})` and is therefore
independent of the exponential difficulty of certifying the optimum.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

#: Defaults matching the configuration used for the verification tables shipped with this
#: repository: 2 ** 12 parallel replicas and 3000 integration steps.
DEFAULT_NUM_REPLICAS = 2**12
DEFAULT_NUM_STEPS = 3000

#: Time steps tried by :func:`solve` when ``dt=None`` requests automatic tuning.
DT_CANDIDATES = (0.25, 0.5, 0.75, 1.0, 1.25)


@dataclasses.dataclass
class Instance:
    """An Ising instance in the convention documented in the module docstring."""

    h: np.ndarray
    W: np.ndarray

    @property
    def num_variables(self) -> int:
        return self.h.shape[0]

    def energy(self, state: np.ndarray) -> np.ndarray:
        """Energy of one state of shape ``(N,)`` or of a batch of shape ``(N, R)``.

        Always evaluated in ``float64``, so the result is a from-scratch reference value
        rather than an incrementally accumulated one.
        """
        s = np.asarray(state, dtype=np.float64)
        if s.ndim == 1:
            return float(self.h @ s + 0.5 * s @ (self.W @ s))
        return self.h @ s + 0.5 * np.einsum("ir,ir->r", s, self.W @ s)


def load_instance(path) -> Instance:
    """Read an instance from the COO text format written by ``bf.py``."""
    rows, cols, values = [], [], []
    with open(path) as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            i, j, value = line.split()
            rows.append(int(i))
            cols.append(int(j))
            values.append(float(value))

    num_variables = max(max(rows), max(cols)) + 1
    h = np.zeros(num_variables, dtype=np.float64)
    W = np.zeros((num_variables, num_variables), dtype=np.float64)
    for i, j, value in zip(rows, cols, values):
        if i == j:
            h[i] += value
        else:
            W[i, j] += value
            W[j, i] += value
    return Instance(h=h, W=W)


def coupling_scale(instance: Instance) -> float:
    """Positional scale ``c0`` of the coupling term.

    Follows the usual simulated-bifurcation choice of making the coupling term comparable
    to the pump term, i.e. ``c0 = 0.5 / (sqrt(N) * std(J))`` evaluated over the
    off-diagonal couplings.
    """
    num_variables = instance.num_variables
    off_diagonal = instance.W[~np.eye(num_variables, dtype=bool)]
    scale = float(np.std(off_diagonal))
    if scale == 0.0:
        return 1.0
    return 0.5 / (np.sqrt(num_variables) * scale)


@dataclasses.dataclass
class Result:
    """Outcome of a dSB run."""

    state: np.ndarray
    energy: float
    num_replicas: int
    num_steps: int
    dt: float
    time_in_seconds: float
    energies: np.ndarray

    @property
    def num_optimal_replicas(self) -> int:
        """How many replicas ended in a state of the best energy found."""
        return int(np.count_nonzero(self.energies <= self.energy + 1e-9))


def _integrate(
    instance: Instance,
    num_replicas: int,
    num_steps: int,
    dt: float,
    c0: float,
    rng: np.random.Generator,
    dtype=np.float64,
) -> np.ndarray:
    """Run the dSB dynamics and return the final spin configurations, shape ``(N, R)``."""
    num_variables = instance.num_variables
    W = instance.W.astype(dtype)
    h = instance.h.astype(dtype).reshape(-1, 1)

    x = rng.uniform(-0.1, 0.1, size=(num_variables, num_replicas)).astype(dtype)
    y = rng.uniform(-0.1, 0.1, size=(num_variables, num_replicas)).astype(dtype)

    a0 = dtype(1.0)
    for step in range(num_steps):
        pump = a0 * dtype((step + 1) / num_steps)
        signs = np.sign(x)
        np.copyto(signs, dtype(1.0), where=signs == 0)
        y += dt * (-(a0 - pump) * x - c0 * (W @ signs + h))
        x += dt * a0 * y
        at_wall = np.abs(x) > 1.0
        if at_wall.any():
            x[at_wall] = np.sign(x[at_wall])
            y[at_wall] = 0.0

    return np.where(x >= 0, 1, -1).astype(np.int8)


def solve(
    instance: Instance,
    num_replicas: int = DEFAULT_NUM_REPLICAS,
    num_steps: int = DEFAULT_NUM_STEPS,
    dt: typing.Optional[float] = None,
    c0: typing.Optional[float] = None,
    seed: int = 42,
    dtype=np.float64,
) -> Result:
    """Minimize an Ising instance with the discrete simulated bifurcation dynamics.

    :param instance: instance to solve, as returned by :func:`load_instance`.
    :param num_replicas: number of replicas integrated in parallel.
    :param num_steps: number of integration steps.
    :param dt: integration time step. When ``None`` (the default), a short probe run is used
        to pick the best candidate from :data:`DT_CANDIDATES`, mirroring the automatic
        time-step tuning of the GPU solver used for the shipped verification tables.
    :param c0: scale of the coupling term; defaults to :func:`coupling_scale`.
    :param seed: seed of the random number generator used for the initial conditions.
    :param dtype: precision of the dynamics. The reported energies are always recomputed in
        ``float64`` from the returned configurations.
    :returns: a :class:`Result` holding the best configuration found.
    """
    from time import perf_counter

    if c0 is None:
        c0 = coupling_scale(instance)

    start = perf_counter()

    if dt is None:
        probe_replicas = max(64, num_replicas // 16)
        probe_steps = max(200, num_steps // 4)
        best_dt, best_probe = None, np.inf
        for candidate in DT_CANDIDATES:
            states = _integrate(
                instance,
                probe_replicas,
                probe_steps,
                candidate,
                c0,
                np.random.default_rng(seed),
                dtype,
            )
            energy = float(np.min(instance.energy(states)))
            if energy < best_probe:
                best_dt, best_probe = candidate, energy
        dt = best_dt

    states = _integrate(
        instance, num_replicas, num_steps, dt, c0, np.random.default_rng(seed), dtype
    )
    energies = instance.energy(states)
    best = int(np.argmin(energies))
    elapsed = perf_counter() - start

    return Result(
        state=states[:, best].copy(),
        energy=float(energies[best]),
        num_replicas=num_replicas,
        num_steps=num_steps,
        dt=float(dt),
        time_in_seconds=elapsed,
        energies=energies,
    )
