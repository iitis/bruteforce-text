import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent.parent / "omni-bench" / "results"
OUT = Path(__file__).resolve().parent.parent / "distributed.pdf"

N_MIN = 40


def load_bf():
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        with open(p) as fd:
            d = json.load(fd)
        if d["num_variables"] >= N_MIN:
            rows.append((d["num_variables"], d["solve_time_in_seconds"]))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows]


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
    })

    bf_n, bf_t = load_bf()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        bf_n,
        bf_t,
        marker="o",
        linestyle="--",
        color="red",
        label=r"Brute-force ($8\times$ NVIDIA H100, distributed)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("System size $N$")
    ax.set_ylabel("Solver time [s]")
    ax.set_xticks(bf_n)

    references = [
        (1.0, "1 second"),
        (10.0, "10 seconds"),
        (60.0, "1 minute"),
        (3600.0, "1 hour"),
        (86400.0, "1 day"),
        (3 * 86400.0, "3 days"),
    ]
    for y, lbl in references:
        ax.axhline(y, linestyle="--", color="black", linewidth=0.6, alpha=0.7)
        ax.text(bf_n[-1] - 0.3, y * 1.25, lbl, fontsize=10, ha="right")

    ax.set_ylim(0.5, 3e5)
    ax.set_xlim(39, 59)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
