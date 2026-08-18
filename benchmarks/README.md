# Benchmarks and verification

Everything needed to reproduce the figures, tables and verification claims of the manuscript:
the instances, the raw results, the solver drivers, the experiment drivers, and two CPU
heuristics used as independent checks.

```
benchmarks/
├── bench_common.py             shared helpers: environment descriptors, Ray topology, instances
├── bf.py                       the original benchmark sweep (Fig. 1)
├── plot_distributed.py         renders Fig. 1 from results/
├── sbm.py                      CPU discrete simulated bifurcation solver
├── verify_sbm.py               brute force vs heuristic verification tables
├── exp_strong_scaling.py       E1  strong scaling over GPU count and node layout
├── exp_weak_scaling.py         E2  weak scaling at constant work per GPU
├── exp_controller_cost.py      E3  controller cost vs number of subproblems (CPU only)
├── exp_precision.py            E4  float32 against float64
├── exp_qubo.py                 E5  QUBO instances alongside Ising ones
├── exp_instance_families.py    E6  certified verification across instance families
├── exp_boundary_cost.py       E7  end-to-end cost of one Ray node boundary
├── exp_multi_solver.py        E8  dSB and simulated annealing against certified optima
├── gen_wishart.py              Wishart planted instances with a closed-form optimum
├── exp_wishart.py              E9  a family the heuristic fails; certification decides
├── scripts/                    launchers: one per experiment, plus cluster control
├── instances/                  instance files in COO format
└── results/                    raw results, one directory per experiment
```

## Launchers

`scripts/` wraps everything below so that a full revision run is a handful of commands. They
assume a two-node cluster and are **always started from the head node**; the worker is driven
over SSH. Defaults live in `scripts/config.sh` and every one of them can be overridden from
the environment:

| Variable | Default | Meaning |
|---|---|---|
| `HEAD_IP` | `10.40.40.105` | head node, where the scripts are launched |
| `WORKER_IP` | `10.40.40.106` | worker node, started over SSH |
| `GPUS_PER_NODE` | `4` | GPUs per node; topologies are masked down from this |
| `CONDA_ENV` | `omnisolver-bruteforce-bench` | environment activated on both nodes |
| `TOPOLOGIES` | `1x1 1x2 2x1 1x4 2x2 2x4` | series used by the scaling experiments |
| `SKIP_EXISTING` | `1` | skip a point whose result file already exists |

Prerequisites: passwordless SSH from the head to the worker, and the same environment with the
plugin installed on **both** nodes — Ray ships no code, so a worker without the package cannot
solve a subproblem. `scripts/preflight.sh` checks all of that, and compares the two nodes'
CUDA, driver and package versions, before any GPU time is spent.

```shell
cd /path/to/bruteforce-text

./benchmarks/scripts/preflight.sh              # verify both nodes, ~10 s
./benchmarks/scripts/run_all.sh                # every experiment, ~9-10 h; the Fig. 1 sweep is left out
./benchmarks/scripts/run_all.sh --cpu-only     # only what needs no GPU
```

Individually, cheapest first:

```shell
./benchmarks/scripts/run_verification.sh          # brute force vs CPU heuristic    ~10 min, no GPU
./benchmarks/scripts/run_e8_multi_solver.sh       # dSB + SA over 20 instances       ~8 min, no GPU
./benchmarks/scripts/run_e6_instance_families.sh  # E6 families vs certified optima ~18 min, 1 GPU
./benchmarks/scripts/run_e9_wishart.sh            # E9 Wishart: heuristic fails     ~40 min, 1 GPU
./benchmarks/scripts/run_e3_controller_cost.sh    # E3 controller cost to 2^16      ~30 min, no GPU
./benchmarks/scripts/run_e7_boundary_cost.sh      # E7 end-to-end node-boundary cost  ~5 min, 2 nodes
./benchmarks/scripts/run_e4_precision.sh          # E4 float32 vs float64            ~6 min, 1 GPU
./benchmarks/scripts/run_e5_qubo.sh               # E5 QUBO alongside Ising          ~1.6 h, 1 GPU
./benchmarks/scripts/run_e1_strong_scaling.sh     # E1 all six topologies            ~2 h
./benchmarks/scripts/run_e2_weak_scaling.sh       # E2 all six topologies            ~4 h
./benchmarks/scripts/run_manuscript_figure.sh     # Fig. 1 sweep                     see below
```

Times are derived from the measured single-GPU point at *N* = 50 (2112 s); `run_all.sh`
without `--with-figure` therefore takes **roughly 9 to 10 hours**, dominated by E2 and E1.

Each writes a timestamped log to `benchmarks/logs/` and restarts or stops the Ray cluster as
the experiment requires. Scaling launchers skip completed points, and E6--E8 create
timestamped, non-overwriting evidence files. The short E3--E5 drivers and verification driver
update their documented canonical outputs; copy those files first when retaining repeated runs.
`run_e1_*` and `run_e2_*` print a speedup/efficiency summary at the end.

The Fig. 1 sweep is the only multi-day item, and only for points that are missing: with the
published results in place it skips every size and just redraws the figure in seconds. It costs
days only if `results/` has been emptied, since *N* = 58 alone is ≈19 h and *N* = 60 ≈3.15 days.
To redraw without touching the solver at all:

```shell
./benchmarks/scripts/run_manuscript_figure.sh --plot-only
```

To regenerate data from scratch, run it under `tmux` and cap the size:

```shell
tmux new -s figure
./benchmarks/scripts/run_manuscript_figure.sh --max-size 54
```

The cluster can also be driven by hand, which is what the experiment scripts do internally:

```shell
./benchmarks/scripts/ray_cluster.sh start 2x1   # 1 GPU on each of the two nodes
./benchmarks/scripts/ray_cluster.sh status      # detected topology, as JSON
./benchmarks/scripts/ray_cluster.sh stop
```

## Setup

```shell
conda env create -f benchmarks/environment.yml
conda activate omnisolver-bruteforce-bench
export CUDAHOME=/usr/local/cuda
python -m pip install "omnisolver-bruteforce[distributed]"   # or -e ../omnisolver-bruteforce
```

Rendering Fig. 1 additionally requires a system LaTeX installation because the plotting script
uses Matplotlib's `text.usetex` mode; Matplotlib and Pillow are included in the Conda file.

`sbm.py`, `verify_sbm.py`, `exp_multi_solver.py` and `exp_controller_cost.py` need no
GPU. The E7 driver also performs no GPU computation, although its launcher uses the same
two-node Ray deployment as the GPU experiments.

## Reproducing Figure 1

```shell
python benchmarks/bf.py --start 40 --stop 60 --step 2 --sampler-mode distributed --skip-existing
python benchmarks/bf.py --start 40 --stop 54 --step 2 --sampler-mode single-gpu  --skip-existing
python benchmarks/plot_distributed.py
```

Results land in `results/{distributed,single-gpu}/<N>.json` and are skipped if already
present, so the published data is never silently overwritten.

## Ray topologies

The scaling experiments distinguish *how many* GPUs are used from *how they are spread over
nodes*. Ray cannot rearrange a running cluster, so the topology is fixed by how `ray start`
is invoked; the experiment scripts detect it, refuse to run if it does not match the
`--topology` label they were given, and record it in the output.

Label `NxM` means N nodes with M GPUs each.

```shell
# 1x1 - one GPU on one node
ray stop --force; CUDA_VISIBLE_DEVICES=0 ray start --head --port=6379 --num-gpus=1

# 1x2 - two GPUs on one node
ray stop --force; CUDA_VISIBLE_DEVICES=0,1 ray start --head --port=6379 --num-gpus=2

# 1x4 - four GPUs on one node
ray stop --force; CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --port=6379 --num-gpus=4

# 2x1 - one GPU on each of two nodes
#   head:   ray stop --force; CUDA_VISIBLE_DEVICES=0 ray start --head --port=6379 --num-gpus=1
#   worker: ray stop --force; CUDA_VISIBLE_DEVICES=0 ray start --address='<HEAD_IP>:6379' --num-gpus=1

# 2x2 - two GPUs on each of two nodes
#   head:   ray stop --force; CUDA_VISIBLE_DEVICES=0,1 ray start --head --port=6379 --num-gpus=2
#   worker: ray stop --force; CUDA_VISIBLE_DEVICES=0,1 ray start --address='<HEAD_IP>:6379' --num-gpus=2

# 2x4 - the configuration behind Fig. 1
#   head:   ray stop --force; ray start --head --port=6379 --num-gpus=4
#   worker: ray stop --force; ray start --address='<HEAD_IP>:6379' --num-gpus=4
```

`1x2` against `2x1`, and `1x4` against `2x2`, hold the GPU count fixed while crossing a node
boundary. That difference is the closest available measurement of the multi-node behavior the
reviewers asked about on a two-node cluster.

## Experiments

Each revision-experiment driver writes to `results/<experiment>/` and stamps its output with
the Python, CUDA, driver, GPU and package versions it ran with. The legacy Figure 1 files
predate these descriptors and are retained unchanged.

### E1 - strong scaling (reviewers 1, 2, 3)

Fixed size, growing GPU count; gives speedup and parallel efficiency, and separates GPU count
from node layout.

```shell
python benchmarks/exp_strong_scaling.py --size 50 --single-gpu       # reference point
# then, on each topology in turn:
python benchmarks/exp_strong_scaling.py --size 50 --topology 1x1
python benchmarks/exp_strong_scaling.py --size 50 --topology 1x2
python benchmarks/exp_strong_scaling.py --size 50 --topology 2x1
python benchmarks/exp_strong_scaling.py --size 50 --topology 1x4
python benchmarks/exp_strong_scaling.py --size 50 --topology 2x2
python benchmarks/exp_strong_scaling.py --size 50 --topology 2x4
```

Expect roughly 2 hours in total at N = 50 on H100-class GPUs: the series is dominated by its
own reference points, since the single-GPU run and the `1x1` run each take about 35 minutes
while `2x4` takes under 5. The `1x1` distributed point is worth having next to `--single-gpu`:
the difference between them is pure Ray overhead.

### E2 - weak scaling (reviewer 2)

Grows the problem with the GPU count so the work per GPU stays at 2^50 configurations. Ideal
weak scaling keeps the wall-clock time flat.

```shell
python benchmarks/exp_weak_scaling.py --single-gpu          # N=50, k=0
python benchmarks/exp_weak_scaling.py --topology 1x2        # N=51, k=1
python benchmarks/exp_weak_scaling.py --topology 2x1        # N=51, k=1
python benchmarks/exp_weak_scaling.py --topology 1x4        # N=52, k=2
python benchmarks/exp_weak_scaling.py --topology 2x2        # N=52, k=2
python benchmarks/exp_weak_scaling.py --topology 2x4        # N=53, k=3
```

Roughly 40 minutes per point at 2^50 configurations per GPU.

The top-level E1/E2 JSON files are the second measurement pass quoted in the manuscript. The
first H100 pass remains in repository history at commit `36a318c`; comparing the two gives a
maximum change of 0.08% for single-node layouts (0.02% in strong scaling) and 1.5% for
two-node layouts.

### E3 - controller cost (reviewers 2, 3)

Answers "would the merge limit the algorithm at 2^16 workers?" **without** 2^16 GPUs: the
number of subproblems is set by `num_fixed_vars`, not by the device count, so the controller
can be driven on a CPU with the partial results a worker would have returned. Compares the
sequential merge used up to release 0.0.5 against the hierarchical one used now, and both
against the floor set by Ray's task scheduling.

```shell
python benchmarks/exp_controller_cost.py --max-k 14          # minutes, no GPU
python benchmarks/exp_controller_cost.py --max-k 16          # 2^16 subproblems, needs RAM
```

Note what this does **not** measure. Running on one machine, every partial result is fetched
from the local object store and every task is scheduled by the local raylet, so the sweep
contains no network transfer at all. Its numbers bound the local component of the controller
cost from below; they are not an estimate of what a real allocation of that size would pay.
The manuscript uses the H100-server production file `results/controller_cost/k3-16.json`,
which is the only sweep retained in that directory.

### E7 - end-to-end cost of a node boundary (reviewers 2, 3)

Measures the aggregate end-to-end term E3 omits on the available two-node cluster. For each
payload and batch size, matched tasks receive the same request and construct the same
`SampleSet`; one returns it and the other returns `None`. The timer begins before task submission
and ends after `ray.get` has materialized every result. Local, remote and mixed placements are
measured, task order is alternated, and every repetition is retained.

```shell
./benchmarks/scripts/run_e7_boundary_cost.sh --topology 2x1
```

First run `--smoke`; it exercises every payload and placement with one small batch. The
production defaults sweep 1, 100 and 1000 returned states (approximately 1, 50 and 500 kB for
$N=60$) at batch sizes 16 and 64, retain five repetitions, and write a timestamped
non-overwriting result. The experiment uses Ray's ordinary task-return path and does not force
or classify a storage mechanism. Consequently it reports the combined cost of serialization,
delivery and materialization rather than standalone network bandwidth. It describes only the
tested two-node loads and is not extrapolated to thousands of workers.

The shipped production record is
`results/boundary_cost/2x1_production_20260816T122920_076038Z.json`. For batches of sixteen,
the paired remote-minus-local payload-return effect is 196.45 microseconds per task for the
50,165-byte payload and 4.404 milliseconds per task for the 496,575-byte payload. The
1,061-byte point and the larger batches are scheduling-noisy; consult the retained raw
repetitions rather than fitting a bandwidth model to these data.

### E4 - single against double precision (reviewer 2)

```shell
python benchmarks/exp_precision.py --sizes 40 42 44 46
```

Note that `float64` is a different numerical path, not merely a slower one: the compensated
updates and periodic re-anchoring are enabled only for `float32`, and only from 40 variables
per kernel upwards. On the reported H100 run, excluding the N=40 warm-up outlier, `float64`
was only 2.9--3.3% slower; consumer GPUs with much lower FP64 throughput can behave
differently.

### E5 - QUBO instances (reviewer 3)

```shell
python benchmarks/exp_qubo.py --sizes 40 42 44 46 48 50
```

Loads the same coefficient files once as SPIN and once as BINARY. The energies differ, since
the same coefficients define different objectives; the runtimes should not, because both are
solved by the identical QUBO code path.

### E6 - instance families (reviewer 1)

```shell
python benchmarks/exp_instance_families.py --size 44 --replicas 5
```

Sweeps uniform, bimodal, Gaussian and sparse couplings, and asks for each instance whether the
heuristic recovers the certified optimum. Roughly 11 minutes at N = 44 for 20 instances on an
H100; the cost doubles per added variable, so N = 48 is about 3 hours. Each invocation creates
a timestamped result by default, preserving the shipped canonical source used by E8.

### E8 - targeted multi-solver comparison (reviewer 1)

```shell
./benchmarks/scripts/run_e8_multi_solver.sh
```

Reads the immutable E6 certificates and reruns both the shipped dSB heuristic and an
algorithmically distinct simulated-annealing solver on the same host and the same twenty
certified $N=44$ instances. dSB uses 4096 replicas, 3000 integration steps and seed 42. SA uses
4096 reads, 3000 sweeps, a geometric inverse-temperature schedule, randomized update order and
deterministic seeds 420000 through 420019. The result records each selected state and its
independently recomputed energy, every SA seed run, instance/source/driver SHA-256 digests,
effective schedule, timings, environment and family summary. The dSB and SA budgets are
documented separately; a dSB integration step and an SA sweep are not treated as equal units of
work.

The shipped H100-server production record is
`results/multi_solver/run_20260816_125511.json`. Both methods reach all 20 certified optima;
their overall median wall times under the stated, method-specific budgets are 9.21 seconds for
dSB and 10.80 seconds for SA.

### E9 - Wishart planted family (reviewer 1)

```shell
./benchmarks/scripts/run_e9_wishart.sh
```

The complement to E6/E8: a family on which the heuristic *fails*. Wishart planted instances
(Hamze et al., Phys. Rev. E 101, 052102 (2020)) have a first-order landscape whose ruggedness
is tuned by `alpha = M/N`; small alpha defeats the dSB heuristic at N = 40, a size the
brute-force sampler certifies in seconds. Every instance carries a closed-form ground-state
energy `E_0 = -tr(W W^T)/(2N)`, attained by the planted state, so each brute-force certificate
is checked against an analytic value that is independent of both solvers, and an energy below
it halts the run. The experiment runs two paired arms: *planted* (closed-form `E_0`, attained
by the planted state) and *unplanted* (the projection is skipped, so the optimum is unknown a
priori and exhaustive search is the only source of truth; `-tr(W W^T)/(2N)` survives as a
strict lower bound that every certificate must clear). The two arms share their random draws
per `(alpha, replica)`, so removing the planting is the only difference. Instances are
deterministic in `(seed, size, alpha, replica)` (`gen_wishart.py`), written once and
re-derived rather than rewritten on later runs. dSB uses the same budgets as E6/E8: 4096
replicas, 3000 integration steps, seed 42, float64 dynamics, automatic time-step selection.
Scoring is on energy; the zero field makes the Z2 mirror exactly degenerate, so Hamming
distances are informational. Each invocation writes a timestamped, non-overwriting record
under `results/wishart/`. With `--ensembles planted --skip-bruteforce` the planted arm is
scored against the analytic `E_0`, so its family-level conclusion can be re-checked without
any GPU (the unplanted arm has no GPU-free score by construction). `plot_wishart.py` renders
the success-fraction-versus-alpha figure, one series per ensemble, from the newest record (or
a given one) into the manuscript directory.

## Verification against the heuristic

`sbm.py` implements the discrete simulated bifurcation dynamics (dSB) in NumPy: all replicas
are integrated at once, so a step is one dense `(N, N) @ (N, R)` product and the total cost is
`O(N^2 * replicas * steps)` — independent of the exponential cost of *certifying* the optimum.
N = 60 with 4096 replicas and 3000 steps takes well under a minute on a CPU.

```shell
python benchmarks/verify_sbm.py --check
```

This recomputes every stored brute-force energy from scratch in `float64`, runs the heuristic
on the same instances, and writes the `bf_sbm_verification_*` tables. With `--check` it also
acts as a regression gate, exiting non-zero if the heuristic failed to reach the certified
optimum on any instance.

It reaches the certified optimum on every stored brute-force result — all twenty runs,
N = 38…60 — returning configurations **identical** to the certified ground states (Hamming
distance 0), with energies agreeing to `float64` round-off.

## Conventions

Instances are `i j value` triples, zero-based, with `i <= j`. A diagonal entry is a linear
bias, an off-diagonal entry a coupling, and the energy of a spin configuration is
`E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j` — exactly how `dimod` reads the same file with
`vartype="SPIN"`.

The shipped Fig. 1 instance files are the canonical inputs and `bf.py` preserves them by
default. The `--seed` option affects files that are actually generated. When files are
deliberately regenerated, `bf.py` draws sequentially across the requested sweep, so a file's
content depends
on which other sizes are generated in the same invocation. `bench_common.generate_instance`
instead seeds from `(seed, size, family, replica)`, so each file is reproducible on its own; it
never overwrites an existing file, because the instances behind the published results are part
of the record.
