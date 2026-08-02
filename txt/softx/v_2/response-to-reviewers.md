# Response to reviewers

**Manuscript:** *Omnisolver: An extensible interface to Ising spin–glass and QUBO
solvers — a numerically stabilized, distributed GPU exhaustive-search plugin*
(SoftwareX, Original Software Publication update)

**Authors:** K. Jałowiecki, J. Pawłowski, B. Gardas, Ł. Pawela

---

> **INTERNAL — REMOVE BEFORE SUBMISSION.**
> Blocks marked `[TODO]` depend on work that is not finished yet. Each one states
> exactly what has to exist before the surrounding paragraph is truthful.
>
> Updated 2026-08-02, after the plugin-repository work: the code side of **R3.6**, **R3.7**
> and **R2.5** is done, so what remains for those three is *publishing* (tag `0.0.6`,
> deploy the documentation) and *one measurement* (E4). Still open: the public location of
> the data, experiments E1–E6, and the five wording corrections in the manuscript that the
> code changes force (see `REVIEW_ANALYSIS.md` §6, Etap 6). Statuses are tracked in
> `REVIEW_ANALYSIS.md` §0.

---

We thank all three reviewers for a careful and constructive reading. The revision
changes the framing of the contribution substantially, adds three tables, five new
paragraphs and a corrected precision analysis, and fixes several statements that did
not survive our own re-check of the underlying data. We are particularly grateful to
Reviewer 3 for pointing out that the plugin was already announced in the original
publication: acting on that comment led us to rewrite the contribution claim from
scratch, which we believe makes the paper both more accurate and easier to assess.

Two corrections we made on our own initiative deserve to be stated up front, since
they change numbers that appeared in the submitted version:

1. **The claim of near-ideal speedup "from N ≳ 44" was wrong.** While computing the
   parallel-efficiency table requested by Reviewer 1, we found that at *N* = 44 the
   measured efficiency is 43%, not ≈100%. The correct statement — now used throughout —
   is that the two samplers cross between *N* = 40 and *N* = 42, efficiency exceeds 90%
   from *N* = 48, and the ideal 8× speedup is attained from *N* = 52 onwards.
2. **The precision paragraph quoted a range not covered by our data.** The submitted
   text reported deviations and Hamming distances "at *N* = 56, 58, and 60"; the
   verification tables in fact cover *N* = 40…54 (single-GPU) and *N* = 60
   (distributed). The paragraph has been rewritten around the instances that are actually
   verified — and, with the CPU heuristic described under R2.1, the verification now covers
   all twenty stored runs, *N* = 38…60, so the claim ends up broader than the one withdrawn.

All changes are confined to the revised manuscript; a `latexdiff` against the submitted
version is provided.

---

## Reviewer 1

### R1.1 — "The paper measures up to 8 H100 GPUs, but claims about scaling to much larger allocations are back-of-the-envelope projections. The authors also note that controller merge bandwidth, Ray overhead, and kernel launch latency may limit scaling."

**We agree, and we have reduced both the size and the prominence of the projection.**

Three changes:

- The projection target is now **2^14 = 16,384 GPUs** rather than 2^16. We checked our
  original phrasing and it was simply wrong: 2^16 = 65,536 accelerators exceeds the
  accelerator count of any system in operation, so calling it "an allocation within
  reach of leadership-class systems" was not defensible. The revised paragraph gives
  *N* = 60 in ≈2.2 min and *N* = 64 in ≈35 min at 2^14 GPUs, with a one-month budget
  reaching *N* ≲ 74. 2^16 now appears only as an explicitly hypothetical bound, with the
  observation that quadrupling the resources buys two variables — which we think conveys
  the real message better than the original number did.
- The projection is confined to a single clearly labelled paragraph (*Scaling outlook on
  larger allocations*) and no longer appears in the abstract or the conclusions.
- It is now **preceded** by a new paragraph, *Controller cost and the limits of the
  decomposition*, which states which component would actually break first and why (see
  R2.2), and it closes by pointing back to that caveat.

We have also added an explicit limitation (item (v) of *Limitations and intended scope*)
stating that eight GPUs across two nodes is the extent of the hardware available to us, so
that while the controller has been measured directly up to 2^16 subproblems, the behavior of
the *search* phase at such allocations remains an extrapolation from eight devices.

The projection is now also bounded from below by a measurement rather than only by an
argument. Table 4 of the revision reports the controller cost for up to 2^16 subproblems,
measured on a CPU: at that scale the dispatch and merge together cost about 6 s, so for the
projections above they inflate the wall-clock time by a few percent rather than dominating it.

> `[TODO]` **E3 is done** (`benchmarks/exp_controller_cost.py`, results in
> `benchmarks/results/controller_cost/`). What is still an extrapolation is the *search* phase
> beyond eight GPUs; experiment **E1** would add the measured efficiency-versus-GPU-count curve
> that makes the linear-scaling assumption behind the projection explicit.

### R1.2 — "The paper positions the plugin as a ground-truth oracle and briefly checks results against SBM, but it does not provide a broad benchmark against multiple exact or heuristic solvers across different instance families."

**We have added a paragraph that addresses this directly, and we would like to explain
why part of the requested benchmark would not be informative for this particular kind of
solver.**

The runtime of exhaustive search does not depend on the instance family. All 2^N
configurations are enumerated regardless of the distribution of the coefficients, and up
to the negligible bookkeeping of best-state updates the time to solution is a function
of *N* and of the CUDA launch geometry alone. A sweep over coupling distributions would
therefore reproduce Fig. 1 rather than add information. Similarly, a runtime comparison
against other *exact* solvers has a predetermined outcome: any exact method must visit an
exponential number of configurations, and CPU implementations are orders of magnitude
slower at the sizes considered here (as quantified in the predecessor paper, ref. [2]).

What *does* depend on the instance family is the difficulty seen by the heuristics that
this plugin exists to certify. We therefore agree with the substance of the comment while
relocating it: the informative experiment is to repeat the *verification* — does the
heuristic still recover the certified optimum? — across several families (bimodal
*J* ∈ {−1,+1}, Gaussian, sparse). This is now stated in the new paragraph *Instance
families, QUBO instances and other solvers* and listed as item (iv) of the future work in
the conclusions.

On exact solvers we have also made the existing cross-check explicit: the plugin's test
suite compares against `dimod.ExactSolver` on instances small enough for CPU
enumeration, and every returned configuration at benchmark scale is re-verified by a
from-scratch `float64` energy recomputation outside the plugin.

> `[TODO]` If experiment **E6** (three instance families × 5 instances at *N* = 48,
> brute-force vs. heuristic agreement) is run, add the resulting table and change the
> paragraph from "the informative extension is…" to a report of the outcome.

### R1.3 — "The paper claims near-ideal speedup, but does not explicitly calculate or report parallel efficiency."

**Agreed and added — and, as noted above, doing so revealed an error in the submitted
version.**

The revision contains a new table (**Table 3**) and a new paragraph (*Strong scaling and
parallel efficiency*). Both samplers were run on identical instances, so the two curves of
Fig. 1 pair size by size into a strong-scaling measurement. Computed from the raw result
files, the speedup *S* = *t*₁/*t*₈ and efficiency *E* = *S*/8 are:

| *N* | *t*₁ [s] | *t*₈ [s] | *S* | *E* |
|----:|---------:|---------:|----:|----:|
| 40 | 2.74 | 5.68 | 0.48 | 6% |
| 42 | 8.20 | 6.52 | 1.26 | 16% |
| 44 | 32.83 | 9.48 | 3.46 | 43% |
| 46 | 131.54 | 21.90 | 6.01 | 75% |
| 48 | 527.07 | 71.17 | 7.41 | 93% |
| 50 | 2112.29 | 268.90 | 7.86 | 98% |
| 52 | 8458.17 | 1060.68 | 7.97 | 99.7% |
| 54 | 33912.27 | 4235.25 | 8.01 | 100.1% |

This shows that our original statement — full ≈8× speedup "from *N* ≳ 44" — was
incorrect, and we have corrected it in the abstract, in *Architecture*, in *Performance*
and in the caption of Fig. 1. Below the crossover the decomposition is dominated by
fixed overheads; at *N* = 40 the eight-way split is in fact 2.1× *slower* than a single
GPU, which we now state explicitly as the reason the single-GPU sampler remains the right
choice whenever one device suffices.

We report the value at *N* = 54 as measured (*E* = 100.1%) together with a one-sentence
explanation that it lies within the run-to-run variability of these single-run timings
and should be read as ideal scaling rather than superlinear speedup, rather than rounding
it down silently.

---

## Reviewer 2

### R2.1 — "The link to the script containing Simulated Bifurcation Machine (SBM) parameters used for verification currently doesn't work. So, this section should contain the updated link."

**The reviewer is right, and we have changed the arrangement so that this cannot happen
again: the data now travels with the article rather than in a separate repository.**

The instances, the raw per-run results and the verification tables are included in the
article repository under `benchmarks/`:

```
benchmarks/instances/<N>.txt                       # the exact instances used
benchmarks/results/single-gpu/<N>.json             # per-run timings, energies, states
benchmarks/results/distributed/<N>.json
benchmarks/bf_sbm_verification_*_summary.csv       # per-size verification tables
benchmarks/bf_sbm_verification_*_states.csv        # returned configurations
```

The benchmark and plotting scripts in `code/` read from these paths, so Fig. 1 and
Table 3 can be regenerated from a single checkout with no external repository involved.
The manuscript now points at this one location consistently.

Regarding the SBM parameters specifically, we have done two things. First, the revision
states the full configuration inline in the paragraph *Verification against an independent
solver* — a chaotic-variant, discrete simulated-bifurcation solver, with discrete and ternary
update rules, automatic time-step tuning, single-precision arithmetic and Kerr and heating
terms disabled, executed on a single H100 GPU with 2^12 = 4096 parallel replicas and 3000
integration steps, reporting the lowest-energy replica; a parameter list in the paper is more
durable than a pointer to a script.

Second, and more usefully, we **reimplemented the same discrete dynamics from scratch in
NumPy** and ship it as `benchmarks/sbm.py`, driven by `benchmarks/verify_sbm.py`. This removes
the dependency the reviewer's comment exposed: the verification can now be reproduced by
anyone, on a CPU, with nothing but the article repository. Because all replicas are integrated
simultaneously, one step is a single dense (N×N)(N×4096) product, so the whole run costs
O(N² · replicas · steps) — independent of the exponential cost of *certifying* the optimum, and
a matter of seconds even at N = 60.

The reimplementation reproduces the published record and extends it. With the same replica and
step counts it returns configurations **identical** to the certified ground states on all nine
instances the original tables cover, with energies agreeing to 3e-14, and it reaches the
certified optimum on **all twenty** stored brute-force runs, N = 38…60. The verification
paragraph and the precision paragraph have been widened accordingly, and both sets of tables
are shipped: `bf_sbm_verification_*` from the GPU solver and `bf_sbm_cpu_verification_*` from
the CPU one.

> `[TODO]` This answer requires the article repository to be **publicly reachable** at
> the URL given by `\datarepo` in the preamble (or archived with a DOI) at the moment of
> resubmission. Verify before sending. Also decide whether a runnable verification
> harness is shipped alongside the tables; the driver used for these runs depends on a
> solver package we do not distribute, which is why the parameters are given inline
> instead (see `REVIEW_ANALYSIS.md` §4.9).

### R2.2 — "While the code achieves near ideal strong scaling for N ≥ 44, corresponding weak scaling tests are absent … This should help quantify the slowdown associated with the controller merge step … which should ideally scale as the logarithm of processor count … running this algorithm on 2^16 processors could result in an algorithm that is limited by this communication step."

**This is the most useful comment in the review, and our honest answer is that the
reviewer's concern is justified — but for a different reason than assumed.** We have
added a paragraph (*Controller cost and the limits of the decomposition*) that says so
explicitly.

Three points, all now in the manuscript:

1. **The merge does not scale as log P; it scales as O(2^k).** The reviewer's expectation
   of a *C*·log *P* cost presumes a tree reduction. The released implementation performs a
   sequential concatenation of the 2^k partial sample sets on the controller, and
   retrieves them one object reference at a time, so both the merge and the transfer are
   linear in the number of subproblems. We prefer to state this plainly rather than let
   it be discovered: the hierarchical reduction that would restore logarithmic behavior
   is future work, and it is now item (i) of the conclusions rather than a passing remark.
2. **We have defined what the reported times measure.** This was not stated in the
   submitted version. For the single-GPU sampler the timer brackets the CUDA search
   itself, excluding host-side assembly of the QUBO matrix and decoding of the returned
   configurations; for the distributed sampler it brackets subproblem dispatch and the
   collection of all partial results, **excluding the final merge**. Both exclusions are
   polynomial in *N* and negligible at the reported sizes, but they matter for the
   extrapolations, and the reader is entitled to know.
3. **The trade-off in *k* is now made explicit.** At fixed *N*, incrementing *k* halves
   the work per subproblem while doubling the merge, so a break-even point in *k*
   necessarily exists and moves toward smaller *k* as *N* decreases. This is the
   mechanism behind the reviewer's concern about 2^16 processors, and we agree it is the
   component that decides how far the decomposition can be pushed.

We also note the methodological point the reviewer's suggestion makes possible: measuring
the controller does **not** require a large GPU allocation, because the number of
subproblems is set by `num_fixed_vars` and is independent of the number of devices — Ray
simply queues 2^k tasks onto whatever is available. This is now stated in the manuscript
as the natural companion measurement.

On weak scaling specifically, we thank the reviewer for the concrete design. We note
that it requires *k* to grow together with the device count so that the per-GPU work
2^{N−k} stays fixed — which is exactly what the suggested *N* = 50–53 sweep at 1, 2, 4
and 8 GPUs achieves. It is item (ii) of the conclusions.

**We have now measured the controller directly, and the answer is more interesting than
either of us expected.** Because the number of subproblems is set by `num_fixed_vars` and is
independent of the device count, the controller can be exercised on a CPU with the partial
results workers would have returned — no large allocation needed. Table 4 of the revision
reports that sweep up to 2^16 subproblems. The reviewer's expectation of linear growth is
confirmed for the merge itself (we fit O(P^0.9)), but its constant is small: at 2^16
subproblems the whole controller cost is about 6 s, of which the concatenation is 1.6 s.
The dominant term turns out to be Ray's per-task scheduling — dispatching and collecting 2^16
*empty* tasks already costs 4.8 s — and that floor applies to dispatching the subproblems
themselves, so it is a property of distributing the work at all rather than of how results are
merged. We have therefore rewritten the future-work item: the target is the scheduling
overhead, for instance by grouping several fixed assignments per task, not the merge.

We also implemented the hierarchical merge, and report honestly that it is *not* uniformly
better: below roughly 10^4 subproblems it is slower than the direct merge, because it replaces
a cheap local concatenation with several rounds of task scheduling; above that its logarithmic
depth wins. Both strategies ship, the direct one is the default, and the measured crossover is
documented.

> `[TODO — E1 AND E2 STILL MISSING.]`
>
> E3 is done. The GPU-side experiments are not: **E1** (GPU-count sweep with the node-layout
> comparison) and **E2** (weak scaling exactly as the reviewer designed it, N = 50…53 at 1, 2,
> 4 and 8 GPUs) are implemented in `benchmarks/exp_strong_scaling.py` and
> `benchmarks/exp_weak_scaling.py` and need roughly one day of cluster time. Until they are
> run, this section should not claim a weak-scaling result.

### R2.3 — "While I could locate the code metadata table, I couldn't find the corresponding software metadata table."

**Added.** The revision contains the *Current executable software version* table
(S1–S7) as **Table 2**, next to the existing code metadata table. Two entries deserve a
comment:

- **S2**: the PyPI distribution is a source archive only, so the "executables" are
  compiled at installation time against the user's CUDA Toolkit. We state this in the
  table rather than leaving it implicit.
- **S5**: lists the CUDA Toolkit version used in continuous integration (12.5) and the
  one used for the reported measurements (12.4), together with `ray >= 2.9`, which is
  required by the distributed sampler.

> `[TODO]` S6 (link to user manual) points at the framework documentation and is
> contingent on R3.7 below.

### R2.4 — "I couldn't locate `code/bf.py` in the directory containing the repository (C2). This is presumably also available at [omni-bench], but the link doesn't work currently. Also, in Section 2, it is stated that this benchmarking sweep is in C2, but I presume this is not the case as other areas in the manuscript reference [omni-bench]."

**The reviewer identified a genuine internal contradiction, and it is fixed.** The
submitted manuscript attributed the benchmark script to C2 in Section 2 while attributing
it to the article repository in Section 1. Section 2 now reads: the plugin sources and
documentation are in the plugin repository (C2); the benchmark and plotting scripts are
in the article repository under `code/`; and the data they consume ships alongside them
under `benchmarks/`. The paths inside the scripts have been updated accordingly, so no
script refers to an external repository any more.

### R2.5 — "The manuscript could also benefit from providing speedup with float32 numbers for `num_states = 1` by conducting the simulation at the default float64 precision. This will help users … quantify the expected slowdown."

> `[TODO — ONE MEASUREMENT MISSING. Do not send this section as it stands.]`
>
> The code that blocked this is fixed (2026-08-02); what is missing is experiment **E4**:
> single-GPU, `num_states=1`, *N* = 40…46, both `dtype` values, roughly 15 minutes of GPU
> time.
>
> Two defects had to be repaired first, and both are worth mentioning in the final answer
> because they change what a user gets:
> 1. the SPIN→BINARY recursion in `DistributedBruteforceGPUSampler.sample` passed eight
>    positional arguments and silently dropped `dtype`, so SPIN inputs always ran in
>    `float32`. All arguments are now forwarded by keyword, with a regression test that
>    discriminates the two precisions (float32 → 2.2e-7 from the exact optimum,
>    float64 → 3.6e-15);
> 2. more seriously, the CLI's `--dtype` never worked: `gpu.yml` declares it as a string
>    and NumPy resolves both `"float"` and `"double"` to `float64`, so from the command line
>    *both* choices ran in double precision and the stabilized single-precision path was
>    unreachable. `"float"` now means single precision, as documented.
>
> The revision already contains the qualitative half of the answer, in limitation (iii):
> the stabilization applies only to the `float32` path and engages when the size seen by the
> kernel is at least 40 variables, so `float64` is a different numerical path and not merely
> a slower one. Draft answer once the numbers exist: report the measured `float32`/`float64`
> ratio in *Precision of the reported energies*, note that the `num_states > 1` path is
> separate, and point users at the `dtype` argument.

---

## Reviewer 3

### R3.1 — "N is never defined, even though it plays a central role in interpreting the results. It would be helpful to briefly reintroduce Omnisolver and its key parameters in the introduction."

**Both points addressed.**

*N* is now defined at its first substantive use in the abstract, again in the second
paragraph of Section 1, and once more in the caption of Fig. 1, as the number of binary
variables of an instance (equivalently, the number of Ising spins), so that exhaustive
search enumerates all 2^N configurations.

The same paragraph introduces the sampler parameters that previously appeared in the
code listing without explanation: `num_states`, `num_fixed_vars` = *k*, `suffix_size`
(with the meaning that 2^`suffix_size` configurations are resident in the GPU working
set at a time and the search sweeps 2^{*N*−`suffix_size`} such chunks), and the launch
parameters `grid_size`, `block_size`, `num_steps_per_kernel` and
`partial_diff_buffer_depth`. Section 1 also opens with a short reminder of how Omnisolver
plugins are registered and what the framework provides.

### R3.2 — "The omnisolver-bruteforce plugin is already mentioned in the original paper [1]. Therefore, statements such as 'this update introduces omnisolver-bruteforce' … are misleading. The paper should clearly state what is actually introduced by the software update described here."

**The reviewer is correct, and on checking we found the problem to be broader than
stated.** The original publication not only lists the plugin and notes that it can be
GPU-accelerated; it also reports a benchmark of the GPU sampler on 1, 2, 4 and 8 NVIDIA
A100 GPUs, and describes the plugin as usable up to about 50 variables. Our submitted
manuscript therefore claimed *two* things as new that were not: "first-class Omnisolver
plugin" and "distributed multi-GPU execution". We have rewritten the contribution claim
accordingly:

- **Title** changed to "… — a numerically stabilized, distributed GPU exhaustive-search
  plugin" (this also removes the double colon).
- **Abstract** now says the plugin "together with a preliminary multi-GPU benchmark, was
  already announced in the original publication" before listing what this update adds.
- **A new paragraph**, *Relation to the original publication and to the predecessor
  solver*, states the provenance explicitly — the plugin entry and the 1/2/4/8-A100
  benchmark in ref. [1], and the single-GPU CUDA solver of ref. [2] underneath it — and
  concludes that "what the update contributes is therefore not the existence of a
  GPU-accelerated or multi-GPU exhaustive search, but its maturation into a dependable
  certification backend".
- **The contribution list** was rewritten around what is verifiable in the released code:
  (i) a released, versioned, API-stable distributed sampler in place of the unreleased
  code path behind the earlier benchmark; (ii) numerical stabilization of the `float32`
  ground-state path; (iii) 64-bit-safe enumeration bookkeeping; (iv) certification reach
  extended from *N* ≈ 50 to *N* = 60, i.e. three orders of magnitude in enumerated
  configurations, with an explicit strong-scaling analysis; (v) end-to-end verification
  of the returned optima.
- **The conclusions** open with the same framing rather than with "promotes … to a
  first-class Omnisolver plugin".

### R3.3 — "To properly assess the scalability of the distributed sampler, experiments should include more than two nodes … it would be interesting to investigate whether the controller becomes a bottleneck as the number of nodes increases. For these reasons, the scaling outlook also appears somewhat optimistic."

**We agree on all three counts and have made the corresponding changes; on the hardware
we can only be transparent.**

Two nodes with four H100s each, connected by standard Ethernet, is the entire cluster
available to us. Rather than leave that implicit, limitation (v) now states it, and adds
that the behavior of the controller at larger allocations is consequently a projection
based on the structure of the implementation rather than a measurement.

On the substance of the concern, the revision separates the two components: the search
phase involves no inter-worker communication at all (each subproblem is an independent
Ray task that fixes its own variables and returns only the requested samples), so the
single point of centralization is the controller — and, as described under R2.2, its cost
in the current implementation is linear rather than logarithmic in the number of
subproblems. We consider this an admission that the reviewer's suspicion is correct for
the present code, and we have given the hierarchical reduction that fixes it a definite
place in the future work rather than a vague one.

The scaling outlook has been made less optimistic in the concrete sense described under
R1.1: 2^14 instead of 2^16 GPUs, the observation that 2^16 exceeds any existing system,
and an explicit statement that at 2^14 workers the sequential merge is expected to become
the dominant term.

On the specific question of whether the controller becomes a bottleneck, we can now answer
with data rather than with a projection: it does not, at least not first. Table 4 of the
revision measures the controller up to 2^16 subproblems and finds a total cost of about 6 s
there, against Ray's 4.8 s task-scheduling floor for the same number of tasks. The
centralization is real but small; what grows and would eventually bind is per-task scheduling,
which the search dispatch pays as well.

> `[TODO]` The node-layout half of this answer still needs running. The harness distinguishes
> 1x2 from 2x1 and 1x4 from 2x2 — the same GPU count with and without a node boundary — which
> is the closest measurement of multi-node behavior available on a two-node cluster. Until E1
> is run, keep limitation (v) as it stands.

### R3.4 — "QUBO instances are not evaluated in the paper, even though they are an important target use case for the framework."

**The measured path is in fact the QUBO path, and we have made this visible in two
places.**

Both samplers convert a SPIN model to its BINARY equivalent, solve it as a QUBO on the
GPU, and convert the result back. The timings in Fig. 1 are therefore QUBO timings; the
only Ising-specific cost is an O(*N*²) host-side transformation performed outside the
measured region. This is now stated in *Architecture* ("both samplers accept SPIN and
BINARY models…") and again in *Instance families, QUBO instances and other solvers*.

> `[TODO]` Experiment **E5** (~10 min of GPU time: the same sweep with
> `vartype="BINARY"`) would let us show the two curves overlapping instead of arguing
> from the code. The reviewer asked for a result, so this is worth doing even though the
> argument is sound.

### R3.5 — "The paper should also specify which version of the CUDA Toolkit was used. … Which versions of the CUDA Toolkit are supported?"

**Added.** The reported measurements were obtained with **CUDA Toolkit 12.4**; the
plugin is built and tested in continuous integration with **12.5** under Python 3.10 and
3.11. Both numbers now appear in C6 and in S5, and S4 records the target architecture of
the measurements (NVIDIA H100 96 GB, compute capability `sm_90`).

### R3.6 — "The license listed in C3 does not exist in C2."

**The reviewer is right, and the repository has been corrected.** The Apache-2.0 licence
text is now present, declared in the package metadata (`license` field plus the OSI
classifier), included in the distribution, and carried as per-file SPDX headers in every
distributed source file. The manuscript's C1–C3 and S1–S3 refer to the release that
contains it.

> `[TODO — PUBLISHING STEP MISSING.]`
>
> The repository work is done (`LICENSE`, `pyproject.toml` metadata, `MANIFEST.in`, SPDX
> headers in 13 files; `validate-pyproject` passes), but it is **not released yet**. Before
> sending: (1) confirm the copyright holder — the headers currently say "The Omnisolver
> developers", the alternative being the institutions; (2) tag `0.0.6` (the version is
> derived by `setuptools_scm`, so the tag is the release); (3) update C1, C2, S1, S2 and the
> phrase "tagged `0.0.5` and published on PyPI" in the contribution list. Until the tag
> exists, the paragraph above is a promise rather than a fact.

### R3.7 — "The documentation linked in C7 is currently almost empty, contains several 404 errors, and is of little practical use in its current state."

**We agree, and the documentation has been rewritten.** The reviewer's assessment was, if
anything, generous: the single page that existed was a copy of the *framework's* landing
page, describing Omnisolver in general and installing a different plugin in its quickstart.
The plugin now has its own front page and a user guide covering installation, single- and
multi-GPU usage, the meaning and selection of every sampler parameter, the size limits, the
precision and stabilization semantics, and the reported timings; the reference manual is
generated from the docstrings; and the build is exercised in continuous integration.

> `[TODO — PUBLISHING STEP MISSING.]`
>
> Done in the repository: `docs/index.md` rewritten; `docs/userguide.md` written;
> `mkdocs.yml` repaired (dead `plugins.md` nav entry, missing `stylesheets/extra.css`, wrong
> `repo_url`, defunct polyfill.io, fragile `with-pdf`); two configuration errors that broke
> the build on their own fixed (renderer options nested under `docstring_options`, and a
> Sphinx dependency stack declared for an mkdocs project); `.readthedocs.yml` removed —
> it never built, and RTD *cannot* build these docs because mkdocstrings imports the compiled
> CUDA extension — replaced by a GitHub Pages workflow using `mike` on the CUDA-capable
> runner. `mkdocs build --strict` succeeds.
>
> Before sending: run the workflow, put the published URL into C7 and S6
> (`https://euro-hpc-pl.github.io/omnisolver-bruteforce`), and archive or redirect the stale
> Read the Docs project — deleting the config does not remove the site.

### R3.8 — "There are still a few minor typographical and grammatical errors throughout the manuscript. A careful proofreading would help eliminate them."

**Done.** The manuscript has been proofread in full. Specific corrections include the
broken apposition in "certified reference, i.e. lowest-energy, solutions"; "while one
device has both the memory budget and a wall-clock time within reason"; "compute" used
as a noun; the colloquial "is clean already at *N* = 40"; a stray line break and spacing
before the description of the simulated-bifurcation solver; the typography of
"H100 / 96 GB"; the inconsistent capitalization of "Parallel Tempering"; and the double
colon in the title (addressed with R3.2). Spelling has been made consistently American
throughout.

---

## Summary of changes to the manuscript

| Change | Driven by |
|---|---|
| New title; abstract rewritten | R3.2 |
| New paragraph *Relation to the original publication and to the predecessor solver* | R3.2 |
| Contribution list rewritten around released, verifiable additions | R3.2 |
| *N* defined in three places; parameter glossary; Omnisolver reintroduced | R3.1 |
| **New Table 2** — *Current executable software version* (S1–S7) | R2.3 |
| **New Table 3** — strong-scaling speedup and parallel efficiency, plus a new paragraph | R1.3 |
| Speedup claim corrected from "*N* ≳ 44" to "*N* ≳ 52" throughout | own re-check, prompted by R1.3 |
| New paragraph defining what the reported wall-clock times include and exclude | R2.2 |
| New paragraph *Controller cost and the limits of the decomposition* | R2.2, R3.3 |
| *Scaling outlook* rescaled from 2^16 to 2^14 GPUs, with explicit caveats | R1.1, R3.3 |
| New paragraph *Instance families, QUBO instances and other solvers* | R1.2, R3.4 |
| *Precision of the reported energies* rewritten around the data actually available | own re-check |
| New cross-check: single-GPU and distributed samplers agree bit-for-bit at all eight shared sizes | own addition |
| SBM configuration stated in full inline | R2.1 |
| CUDA Toolkit versions in C6/S5; `sm_90` in S4 | R3.5 |
| Section 2 repository attribution corrected; data shipped in `benchmarks/` | R2.1, R2.4 |
| *Limitations* item (i) corrected: *N* − *k* ≤ 64, hence *N* ≤ 64 + *k* | own re-check (removes an internal contradiction with the *N* ≲ 76 projection) |
| *Limitations* items (iii) and (v) added: stabilization threshold; hardware extent | R2.5, R3.3 |
| Full proofread | R3.8 |

## Summary of changes to the software

| Change | Driven by |
|---|---|
| Apache-2.0 `LICENSE`, package licence metadata and per-file SPDX headers | R3.6 |
| Plugin documentation rewritten (front page, user guide, reference manual) and published from CI | R3.7 |
| Continuous integration now actually runs the test suite, on a CUDA 12.4 + 12.5 matrix | R3.5, and the manuscript's validation claim |
| Tests added for the stabilized `float32` path (*N* = 40) and for the distributed sampler | R2.5, R2.2 |
| `dtype` forwarded through the SPIN recursion; `"float"` from the CLI now means single precision | R2.5 |
| `ray` declared as an optional dependency of the distributed sampler | R2.1 |
| Hierarchical merge replacing the sequential controller-side concatenation | R2.2, R3.3 |
| Per-phase timings (`dispatch`/`solve`/`merge`/`total`) reported in `SampleSet.info` | R2.2 |
| `N - k <= 64` and `suffix_size` validated instead of silently truncating | own re-check |
| CLI exposes `num_steps_per_kernel` and `partial_diff_buffer_depth`, so the reported runs are reproducible from the command line | R2.4 |
| README, CLI help text and example instance corrected | R3.8 |
