import json
from pathlib import Path

import matplotlib.pyplot as plt

BENCHMARKS = Path(__file__).resolve().parent
RESULTS = BENCHMARKS / "results"
OUT = BENCHMARKS.parent / "txt" / "softx" / "v_2" / "distributed.pdf"

N_MIN = 40
N_MAX_PLOT = 60


def load_mode(subdir):
    rows = []
    folder = RESULTS / subdir
    for p in sorted(folder.glob("*.json")):
        with open(p) as fd:
            d = json.load(fd)
        if d["num_variables"] >= N_MIN:
            rows.append((d["num_variables"], d["solve_time_in_seconds"]))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows]


def doubling_extrapolation(n_last, t_last, n_max):
    ns = list(range(n_last + 1, n_max + 1))
    ts = []
    t = t_last
    for _ in ns:
        t *= 2.0
        ts.append(t)
    return ns, ts


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
    })

    dist_n, dist_t = load_mode("distributed")
    sg_n, sg_t = load_mode("single-gpu")

    sg_ext_n, sg_ext_t = doubling_extrapolation(sg_n[-1], sg_t[-1], N_MAX_PLOT)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        sg_n,
        sg_t,
        marker="s",
        linestyle="--",
        color="tab:blue",
        label=r"Brute-force (single NVIDIA H100, measured)",
    )
    ax.plot(
        [sg_n[-1]] + sg_ext_n,
        [sg_t[-1]] + sg_ext_t,
        marker="s",
        linestyle=":",
        markerfacecolor="white",
        color="tab:blue",
        label=r"Single-GPU extrapolation ($t_{N+1}=2\,t_N$)",
    )
    ax.plot(
        dist_n,
        dist_t,
        marker="o",
        linestyle="--",
        color="red",
        label=r"Brute-force ($8\times$ NVIDIA H100, distributed)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("System size $N$")
    ax.set_ylabel("Solver time [s]")
    ax.set_xticks(dist_n)

    references = [
        (1.0, "1 second"),
        (10.0, "10 seconds"),
        (60.0, "1 minute"),
        (3600.0, "1 hour"),
        (86400.0, "1 day"),
        (3 * 86400.0, "3 days"),
        (30 * 86400.0, "1 month"),
    ]
    for y, lbl in references:
        ax.axhline(y, linestyle="--", color="black", linewidth=0.6, alpha=0.7)
        ax.text(dist_n[-1] - 0.3, y * 1.25, lbl, fontsize=10, ha="right")

    ax.set_ylim(0.5, 5e6)
    ax.set_xlim(dist_n[0] - 1, dist_n[-1] + 1)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False,
              fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    print("single-GPU extrapolation (N, t [s]):")
    for n, t in zip(sg_ext_n, sg_ext_t):
        print(f"  N={n}  t={t:.1f} s  ({t/86400:.2f} days)")


if __name__ == "__main__":
    main()
