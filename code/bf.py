from omnisolver.bruteforce.gpu import BruteforceGPUSampler
from dimod.serialization import coo
from pathlib import Path
import numpy as np
import argparse
import json

try:
    from omnisolver.bruteforce.gpu.distributed import DistributedBruteforceGPUSampler
except Exception:  # pragma: no cover - optional dependency path (ray)
    DistributedBruteforceGPUSampler = None

# Paths to input instances and results
RESULTS = Path("results")
INSTANCES = Path("instances")

# Json encoder for NumPy objects
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Generation of random instances
def generate(start, stop, step, skip_existing=True):
    INSTANCES.mkdir(parents=True, exist_ok=True)
    for d in range(start, stop + step, step):
        o_path = INSTANCES / f"{d}.txt"
        if skip_existing and o_path.is_file():
            continue
        J = 2*(np.random.rand(d, d) - 0.5)
        with open(o_path, "w") as fd:
            for i in range(d):
                for j in range(i, d):
                    fd.write(f"{i} {j} {J[i, j]}\n")


# Main code launching the brute-force solver
def bench(start, stop, step, sampler_mode="distributed", skip_existing=True):
    RESULTS.mkdir(parents=True, exist_ok=True)
    for d in range(start, stop + step, step):
        i_path = INSTANCES / f"{d}.txt"
        o_path = RESULTS / f"{d}.json"
        if not i_path.is_file():
            continue
        if skip_existing and o_path.is_file():
            continue
        with open(i_path) as fd:
            bqm = coo.load(fd, vartype="SPIN")
        if sampler_mode == "distributed":
            if DistributedBruteforceGPUSampler is None:
                raise RuntimeError(
                    "Distributed sampler is unavailable. Install ray and "
                    "omnisolver-bruteforce distributed dependencies."
                )
            sampler = DistributedBruteforceGPUSampler()
            result = sampler.sample(
                bqm,
                num_states=1,
                num_fixed_vars=3,
                suffix_size=23,
                grid_size=8192,
                block_size=1024,
                partial_diff_buffer_depth=10,
                num_steps_per_kernel=8192,
            )
        else:
            sampler = BruteforceGPUSampler()
            result = sampler.sample(
                bqm,
                num_states=1,
                suffix_size=23,
                grid_size=8192,
                block_size=1024,
                partial_diff_buffer_depth=10,
                num_steps_per_kernel=8192,
            )
        state = result.samples()[0, range(d)]
        result.info.update(
            {
                "num_variables": d,
                "state": state,
                "energy": result.first.energy,
                "sampler_mode": sampler_mode,
            }
        )
        with open(o_path, "w") as fd:
            json.dump(result.info, fd, indent=4, cls=NpEncoder)

def generate_and_bench(start, stop, step, sampler_mode="distributed", skip_existing=True):
    generate(start, stop, step, skip_existing)
    bench(start, stop, step, sampler_mode, skip_existing)

def main():
    parser = argparse.ArgumentParser(
        description="Perform benchmark of bruteforce solver from start to stop with step"
    )

    # Add arguments with default values
    parser.add_argument("--start", type=int, default=60, help="The start value (default: 60)")
    parser.add_argument("--stop", type=int, default=40, help="The stop value (default: 40)")
    parser.add_argument("--step", type=int, default=-2, help="The step size (default: -2)")
    parser.add_argument(
        "--sampler-mode",
        choices=["distributed", "single-gpu"],
        default="distributed",
        help="Sampler backend: distributed Ray-based or single GPU (default: distributed)",
    )
    parser.add_argument('--skip-existing', action='store_true', help='Skip processing if the file exists')
    # Parse the arguments
    args = parser.parse_args()
    generate_and_bench(args.start, args.stop, args.step, args.sampler_mode, args.skip_existing)
    
if __name__ == "__main__":
    main()
