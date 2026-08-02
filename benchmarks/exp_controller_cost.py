"""E3 - cost of the controller as a function of the number of subproblems.

Reviewer 2 asked whether the merge step would limit the algorithm at 2 ** 16 workers. That
question can be answered without 2 ** 16 GPUs: the number of subproblems is set by
``num_fixed_vars`` and is independent of the device count, so the controller can be exercised
on a CPU alone by feeding it the partial results that workers would have returned.

Both merge strategies start from Ray object references, exactly as they do inside the
sampler, so the comparison is apples to apples: the sequential strategy has to pay for
fetching every partial result to the controller, the hierarchical one only for fetching the
single merged result. Four quantities are measured for each k:

``sequential_merge_seconds``
    Fetching all 2 ** k partial results to the controller and running one
    ``concatenate(...).truncate(num_states)`` over them. This is what releases up to 0.0.5
    did, and it is expected to grow linearly in the number of subproblems.

``tree_merge_seconds``
    The same reduction performed by the hierarchy of Ray tasks that the current release uses,
    including the final fetch. Expected to grow with the depth of the hierarchy, but each
    round pays Ray's task-scheduling overhead, which for cheap merges can dominate.

``ray_fanout_seconds``
    Dispatching and awaiting 2 ** k trivial tasks. This is the scheduling floor: no
    distributed merge can be faster than this.

``merge_arithmetic_seconds``
    The concatenation alone, with the partial results already in the controller's memory.
    Separates the cost of the operation from the cost of moving the data.

The merge functions are imported from the plugin rather than reimplemented, so the numbers
describe the shipped code path.

Usage::

    python benchmarks/exp_controller_cost.py --max-k 14
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

import bench_common as common


def make_partial_result(num_variables: int, num_states: int, rng):
    """A stand-in for what one worker returns: num_states samples over all variables."""
    from dimod import SampleSet

    samples = rng.choice([0, 1], size=(num_states, num_variables))
    energies = rng.normal(size=num_states)
    return SampleSet.from_samples(samples, "BINARY", energies)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument(
        "--max-k",
        type=int,
        default=14,
        help="largest number of fixed variables, i.e. 2 ** max_k subproblems (default: 14)",
    )
    parser.add_argument("--num-variables", type=int, default=60)
    parser.add_argument("--num-states", type=int, default=1)
    parser.add_argument("--merge-batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--skip-ray",
        action="store_true",
        help="only measure the controller-side arithmetic, without starting Ray",
    )
    args = parser.parse_args()

    from dimod import concatenate

    rng = np.random.default_rng(common.DEFAULT_SEED)
    template = make_partial_result(args.num_variables, args.num_states, rng)

    ray = None
    merge_partial_results = batched = None
    if not args.skip_ray:
        import ray as ray_module
        from omnisolver.bruteforce.gpu.distributed import _batched, _merge_partial_results

        ray = ray_module
        merge_partial_results, batched = _merge_partial_results, _batched
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        @ray.remote
        def _noop():
            return None

    def timed(callable_, repeats):
        """Median of `repeats` timings, after one discarded warm-up call."""
        callable_()
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            callable_()
            samples.append(perf_counter() - start)
        return float(np.median(samples)), samples

    measurements = []
    for k in range(args.min_k, args.max_k + 1):
        num_subproblems = 2**k
        partials = [template.copy() for _ in range(num_subproblems)]

        arithmetic, arithmetic_all = timed(
            lambda: concatenate(partials).truncate(args.num_states), args.repeats
        )
        entry = {
            "num_fixed_vars": k,
            "num_subproblems": num_subproblems,
            "merge_arithmetic_seconds": arithmetic,
            "merge_arithmetic_seconds_all": arithmetic_all,
        }

        if ray is None:
            entry["sequential_merge_seconds"] = arithmetic
        else:
            # Both strategies start from object references, as they do inside the sampler.
            stored = [ray.put(partial) for partial in partials]

            def sequential_merge():
                fetched = [ray.get(ref) for ref in stored]
                return concatenate(fetched).truncate(args.num_states)

            rounds_seen = []

            def tree_merge():
                refs, rounds = list(stored), 0
                while len(refs) > 1:
                    refs = [
                        merge_partial_results.remote(args.num_states, *batch)
                        for batch in batched(refs, args.merge_batch_size)
                    ]
                    rounds += 1
                rounds_seen.append(rounds)
                return ray.get(refs[0])

            entry["sequential_merge_seconds"], entry["sequential_merge_seconds_all"] = timed(
                sequential_merge, args.repeats
            )
            entry["tree_merge_seconds"], entry["tree_merge_seconds_all"] = timed(
                tree_merge, args.repeats
            )
            entry["num_merge_rounds"] = rounds_seen[-1]
            entry["ray_fanout_seconds"], _ = timed(
                lambda: ray.get([_noop.remote() for _ in range(num_subproblems)]), args.repeats
            )
            del stored

        measurements.append(entry)
        print(
            f"k={k:2d}  P={num_subproblems:6d}  "
            f"arithmetic={entry['merge_arithmetic_seconds']:8.4f}s  "
            f"sequential={entry['sequential_merge_seconds']:8.4f}s"
            + (
                f"  tree={entry['tree_merge_seconds']:8.4f}s "
                f"({entry['num_merge_rounds']}r)  "
                f"fanout={entry['ray_fanout_seconds']:8.4f}s"
                if ray is not None
                else ""
            )
        )
        del partials

    payload = {
        "num_variables": args.num_variables,
        "num_states": args.num_states,
        "merge_batch_size": args.merge_batch_size,
        "repeats": args.repeats,
        "measurements": measurements,
    }
    path = common.write_result("controller_cost", f"k{args.min_k}-{args.max_k}", payload)
    print(f"wrote {path}")

    scaling = [
        (entry["num_subproblems"], entry["sequential_merge_seconds"]) for entry in measurements
    ]
    if len(scaling) >= 2:
        (p0, t0), (p1, t1) = scaling[0], scaling[-1]
        exponent = np.log(t1 / t0) / np.log(p1 / p0)
        print(
            f"sequential merge scales as O(P^{exponent:.2f}) between P={p0} and P={p1} "
            f"(1.00 would be linear, 0.00 constant)"
        )


if __name__ == "__main__":
    main()
