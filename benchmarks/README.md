# Benchmarks and verification

Everything needed to reproduce the figures, tables and verification claims of the manuscript:
the instances, the raw results, the solver drivers, the experiment drivers, and a CPU
implementation of the heuristic used as an independent check.

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
├── instances/                  instance files in COO format
└── results/                    raw results, one directory per experiment
```

## Setup

```shell
conda env create -f benchmarks/environment.yml
conda activate omnisolver-bruteforce-bench
export CUDAHOME=/usr/local/cuda
python -m pip install "omnisolver-bruteforce[distributed]"   # or -e ../omnisolver-bruteforce
```

`sbm.py`, `verify_sbm.py` and `exp_controller_cost.py` need no GPU. Everything else does.

## Reproducing Figure 1

```shell
python benchmarks/bf.py --start 40 --stop 60 --step 2 --sampler-mode distributed --seed 42 --skip-existing
python benchmarks/bf.py --start 40 --stop 54 --step 2 --sampler-mode single-gpu  --seed 42 --skip-existing
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
boundary. That difference is the closest available measurement of the multi-node behaviour the
reviewers asked about on a two-node cluster.

## Experiments

Each script writes to `results/<experiment>/` and stamps its output with the Python, CUDA,
driver, GPU and package versions it ran with.

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

Expect roughly 35 minutes in total at N = 50 on H100-class GPUs. The `1x1` distributed point is
worth having next to `--single-gpu`: the difference between them is pure Ray overhead.

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

### E4 - single against double precision (reviewer 2)

```shell
python benchmarks/exp_precision.py --sizes 40 42 44 46
```

Note that `float64` is a different numerical path, not merely a slower one: the compensated
updates and periodic re-anchoring are enabled only for `float32`, and only from 40 variables
per kernel upwards. Budget for the double-precision runs being several times slower, and much
worse than that on consumer GPUs, whose FP64 throughput is a small fraction of their FP32
throughput.

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
H100; the cost doubles per added variable, so N = 48 is about 3 hours.

## Verification against the heuristic

`sbm.py` implements the discrete simulated bifurcation dynamics (dSB) in NumPy: all replicas
are integrated at once, so a step is one dense `(N, N) @ (N, R)` product and the total cost is
`O(N^2 * replicas * steps)` — independent of the exponential cost of *certifying* the optimum.
N = 60 with 4096 replicas and 3000 steps takes well under a minute on a CPU.

```shell
python benchmarks/verify_sbm.py --check
```

This recomputes every stored brute-force energy from scratch in `float64`, runs the heuristic
on the same instances, writes `bf_sbm_cpu_verification_*` tables, and with `--check` compares
them against the `bf_sbm_verification_*` tables that were produced with a GPU solver. The
shipped tables are never modified.

The CPU implementation reproduces the published record: on all nine instances covered by the
shipped tables it returns configurations **identical** to the certified ground states (Hamming
distance 0), with energies agreeing to `float64` round-off.

## Conventions

Instances are `i j value` triples, zero-based, with `i <= j`. A diagonal entry is a linear
bias, an off-diagonal entry a coupling, and the energy of a spin configuration is
`E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j` — exactly how `dimod` reads the same file with
`vartype="SPIN"`.

`bf.py` draws its instances sequentially across a sweep, so the content of a given file depends
on which other sizes were generated in the same invocation. `bench_common.generate_instance`
instead seeds from `(seed, size, family, replica)`, so each file is reproducible on its own; it
never overwrites an existing file, because the instances behind the published results are part
of the record.
