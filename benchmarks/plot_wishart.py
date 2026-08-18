"""Render the Wishart success-fraction figure from an E9 result file.

For every ensemble and alpha in the record it plots the fraction of instances on which one
dSB run under the documented budgets reached the certified optimum, with a 95% Wilson
interval for the per-instance success probability. The style follows plot_distributed.py,
and the figure is written next to the manuscript as wishart.pdf.

By default the figure shows the unplanted ensemble alone whenever the record contains it:
that is the manuscript figure, since the planted arm tracks it point for point and exists in
the record as the analytic anchor of the certificates. Pass ``--ensembles planted unplanted``
to draw the overlay comparison instead.

Usage::

    python benchmarks/plot_wishart.py                             # manuscript figure
    python benchmarks/plot_wishart.py --ensembles planted unplanted path/to/run.json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

BENCHMARKS = Path(__file__).resolve().parent
RESULTS = BENCHMARKS / "results" / "wishart"
OUT = BENCHMARKS.parent / "txt" / "softx" / "v_2" / "wishart.pdf"

#: Marker conventions follow Fig. 1: blue squares for the first series, red circles for the
#: second. The unplanted markers are open, matching Fig. 1's use of open markers for the
#: series that carries less prior information.
STYLES = {
    "planted": dict(marker="s", color="tab:blue", markerfacecolor="tab:blue"),
    "unplanted": dict(marker="o", color="red", markerfacecolor="white"),
}
DODGE = {"planted": -0.003, "unplanted": 0.003}


def wilson(successes: int, total: int, z: float = 1.959964):
    """95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 1.0
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = (z / denominator) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return p, max(0.0, center - half), min(1.0, center + half)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--ensembles",
        nargs="+",
        choices=["planted", "unplanted"],
        default=None,
        help="series to draw (default: unplanted alone if the record has it, else all)",
    )
    args = parser.parse_args()

    if args.record is not None:
        record_path = args.record
    else:
        candidates = sorted(RESULTS.glob("*.json"))
        if not candidates:
            raise SystemExit(f"no result files under {RESULTS}")
        record_path = candidates[-1]
    with open(record_path) as fd:
        record = json.load(fd)

    available = record.get("ensembles", ["planted"])
    if args.ensembles is not None:
        ensembles = [e for e in args.ensembles if e in available]
    elif "unplanted" in available:
        ensembles = ["unplanted"]
    else:
        ensembles = available
    alphas = record["alphas"]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
    })
    fig, ax = plt.subplots(figsize=(8, 4.2))

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
            xs.append(alpha + (DODGE[ensemble] if len(ensembles) > 1 else 0.0))
            fractions.append(p)
            lows.append(p - low)
            highs.append(high - p)
            labels.append(f"{reached}/{len(rows)}")
            print(f"{ensemble:9s} alpha {alpha:g}: {reached}/{len(rows)} reached, "
                  f"95% Wilson [{low:.3f}, {high:.3f}]")
        style = (
            STYLES["planted"] if len(ensembles) == 1 else STYLES[ensemble]
        )  # a single series wears the house primary regardless of which ensemble it is
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
        if ensemble == ensembles[0]:
            for x, y, text in zip(xs, fractions, labels):
                ax.annotate(
                    text,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, -16) if y > 0.5 else (0, 10),
                    ha="center",
                    fontsize=10,
                )

    ax.set_xlabel(r"Wishart ruggedness parameter $\alpha = M/N$")
    ax.set_ylabel("fraction reaching the certified optimum")
    ax.set_xticks(alphas)
    ax.set_ylim(-0.06, 1.12)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    if len(ensembles) > 1:
        ax.legend(loc="upper left", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT} from {record_path}")


if __name__ == "__main__":
    main()
