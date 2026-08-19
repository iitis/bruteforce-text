"""Render the Wishart figure from the E9 records.

Panel (a): for every ensemble and alpha in the sweep record, the fraction of instances on
which one dSB run under the documented budgets reached the certified optimum, with a 95%
Wilson interval for the per-instance success probability.

Panel (b), when a repeats record is available (an exp_wishart run with
--heuristic-repeats > 1, e.g. E9b): the per-instance time-to-solution ratio

    p_i     = successes_i / repeats_i
    R99(i)  = ceil(ln 0.01 / ln(1 - p_i))          (runs to 99% success)
    D_i     = R99(i) * t_run / T_BF

with t_run the median dSB per-run wall time and T_BF the median brute-force certification
time of that record (certify with --bf-dtype double so the denominator is a from-scratch
float64 ranking). An instance never solved has no finite TTS estimate; it contributes the
smallest D_i its repeat count permits, via the exact one-sided Clopper-Pearson upper bound
p <= 1 - 0.05^(1/n) (more optimistic for the heuristic than Jeffreys), and is drawn as an
open marker. The style follows plot_distributed.py; the figure is written next to the
manuscript as wishart.pdf.

Usage::

    python benchmarks/plot_wishart.py                          # newest sweep record, panel (a)
    python benchmarks/plot_wishart.py SWEEP.json --tts-record REPEATS.json   # both panels
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

BENCHMARKS = Path(__file__).resolve().parent
RESULTS = BENCHMARKS / "results" / "wishart"
OUT = BENCHMARKS.parent / "txt" / "softx" / "v_2" / "wishart.pdf"

#: Marker conventions follow Fig. 1: blue squares, red circles; the unplanted arm is open.
STYLES = {
    "planted": dict(marker="s", color="tab:blue", markerfacecolor="tab:blue"),
    "unplanted": dict(marker="o", color="red", markerfacecolor="white"),
}
DODGE = {"planted": -0.003, "unplanted": 0.003}
ROW = {"planted": 1.0, "unplanted": 0.0}


def wilson(successes: int, total: int, z: float = 1.959964):
    """95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 1.0
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = (z / denominator) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return p, max(0.0, center - half), min(1.0, center + half)


def runs_to_99(p: float) -> float:
    if p >= 1.0:
        return 1.0
    return math.ceil(math.log(0.01) / math.log(1.0 - p))


def panel_success_fraction(ax, record, ensembles):
    alphas = record["alphas"]
    single = len(ensembles) == 1
    for ensemble in ensembles:
        xs, fractions, lows, highs, labels = [], [], [], [], []
        for alpha in alphas:
            rows = [
                m
                for m in record["measurements"]
                if m["alpha"] == alpha and m.get("ensemble", "planted") == ensemble
            ]
            if not rows:
                continue
            reached = sum(m["sbm_reached_optimum"] for m in rows)
            p, low, high = wilson(reached, len(rows))
            xs.append(alpha + (0.0 if single else DODGE[ensemble]))
            fractions.append(p)
            lows.append(p - low)
            highs.append(high - p)
            labels.append(f"{reached}/{len(rows)}")
            print(f"{ensemble:9s} alpha {alpha:g}: {reached}/{len(rows)} reached, "
                  f"95% Wilson [{low:.3f}, {high:.3f}]")
        style = STYLES["planted"] if single else STYLES[ensemble]
        ax.errorbar(
            xs,
            fractions,
            yerr=[lows, highs],
            linestyle="--",
            capsize=3,
            linewidth=1.5,
            markersize=6,
            label=ensemble,
            **style,
        )
        if single:
            for x, y, text in zip(xs, fractions, labels):
                ax.annotate(
                    text,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, -16) if y > 0.5 else (0, 10),
                    ha="center",
                    fontsize=10,
                )
    ax.set_xlabel(r"ruggedness parameter $\alpha = M/N$")
    ax.set_ylabel("fraction reaching the certified optimum")
    ax.set_xticks(alphas)
    ax.set_ylim(-0.06, 1.12)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    if not single:
        ax.legend(loc="upper left", frameon=False, fontsize=10)


def panel_tts(ax, record, rng_seed=0):
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    entries = [m for m in record["measurements"] if "sbm_repeats" in m]
    if not entries:
        raise SystemExit("the --tts-record has no sbm_repeats; run with --heuristic-repeats")
    alphas = sorted({m["alpha"] for m in entries})
    t_run = float(np.median([r["time_in_seconds"] for m in entries for r in m["sbm_repeats"]]))
    t_bf = float(np.median([m["bf_time_seconds"] for m in entries]))
    ensembles = [e for e in ("unplanted", "planted") if any(m["ensemble"] == e for m in entries)]

    for ensemble in ensembles:
        rows = [m for m in entries if m["ensemble"] == ensemble]
        ds, bounds = [], []
        for m in rows:
            n = len(m["sbm_repeats"])
            k = m["sbm_successes"]
            if k > 0:
                d = runs_to_99(k / n) * t_run / t_bf
                bound = False
            else:
                p_upper = 1.0 - 0.05 ** (1.0 / n)  # exact one-sided CP upper bound at k=0
                d = runs_to_99(p_upper) * t_run / t_bf
                bound = True
            ds.append(d)
            bounds.append(bound)
        ds = np.array(ds)
        bounds = np.array(bounds)
        y = ROW[ensemble] + rng.uniform(-0.16, 0.16, size=len(ds))
        style = STYLES[ensemble]
        measured = ~bounds
        if measured.any():
            ax.plot(ds[measured], y[measured], linestyle="none", markersize=6,
                    marker=style["marker"], color=style["color"],
                    markerfacecolor=style["color"])
        if bounds.any():
            ax.plot(ds[bounds], y[bounds], linestyle="none", markersize=6,
                    marker="^", color=style["color"], markerfacecolor="white")
        median = float(np.median(ds))
        ax.plot([median, median], [ROW[ensemble] - 0.26, ROW[ensemble] + 0.26],
                color=style["color"], linewidth=2)
        ax.annotate(rf"median ${median:.0f}\times$",
                    (median, ROW[ensemble] + 0.32), ha="center", fontsize=10,
                    color=style["color"])
        print(f"TTS {ensemble:9s}: median D={median:.1f}x, "
              f"never solved {int(bounds.sum())}/{len(ds)} (lower bounds), "
              f"t_run={t_run:.2f}s, T_BF={t_bf:.2f}s")

    ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] < 1 else 0.05, 1.0,
               color="tab:orange", alpha=0.08)
    ax.axvline(1.0, color="tab:orange", linestyle="--", linewidth=1.2)
    ax.text(0.93, -0.42, "heuristic faster", ha="right", fontsize=10,
            color="tab:orange")
    ax.set_xscale("log")
    ax.set_yticks([ROW[e] for e in ensembles])
    ax.set_yticklabels(ensembles)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel(
        r"$D_i = \mathrm{TTS}_{99}(i)\,/\,T_{\mathrm{BF64}}$"
        + (rf"  at $\alpha={alphas[0]:g}$" if len(alphas) == 1 else "")
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", nargs="?", type=Path, default=None,
                        help="sweep record for panel (a); default: newest without repeats")
    parser.add_argument(
        "--ensembles",
        nargs="+",
        choices=["planted", "unplanted"],
        default=None,
        help="series for panel (a); default: every ensemble the record contains",
    )
    parser.add_argument(
        "--tts-record",
        type=Path,
        default=None,
        help="repeats record for panel (b); default: newest record with sbm_repeats, "
        "panel (b) is omitted if none exists",
    )
    args = parser.parse_args()

    candidates = sorted(RESULTS.glob("*.json"))
    sweep_path, tts_path = args.record, args.tts_record
    if sweep_path is None or tts_path is None:
        for path in reversed(candidates):
            with open(path) as fd:
                r = json.load(fd)
            if r.get("heuristic_repeats", 1) > 1:
                if tts_path is None:
                    tts_path = path
            elif sweep_path is None:
                sweep_path = path
    if sweep_path is None:
        raise SystemExit(f"no sweep record found under {RESULTS}")
    with open(sweep_path) as fd:
        sweep = json.load(fd)

    available = sweep.get("ensembles", ["planted"])
    ensembles = (
        [e for e in args.ensembles if e in available]
        if args.ensembles is not None
        else available
    )

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
    })

    if tts_path is not None:
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]}
        )
        panel_success_fraction(ax_a, sweep, ensembles)
        with open(tts_path) as fd:
            tts = json.load(fd)
        panel_tts(ax_b, tts)
        ax_a.set_title("(a)", loc="left", fontsize=11)
        ax_b.set_title("(b)", loc="left", fontsize=11)
        sources = f"{sweep_path} + {tts_path}"
    else:
        fig, ax_a = plt.subplots(figsize=(8, 4.2))
        panel_success_fraction(ax_a, sweep, ensembles)
        sources = str(sweep_path)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT} from {sources}")


if __name__ == "__main__":
    main()
