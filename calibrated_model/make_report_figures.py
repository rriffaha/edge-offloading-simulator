"""
make_report_figures.py - figures sized for the IEEE two-column interim report.

Same data as make_figures.py, but drawn at one-column width (3.5 in) with no
slide-style titles: the report's captions carry the explanation.

Run from this folder:   python make_report_figures.py
Needs: pip install matplotlib numpy
Reads:  results_calibrated.csv
Writes: figures\\report\\*.png
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

from model import GOOD_NET, OKAY_NET, MEASURED_NET, BAD_NET, EDGE_SERVER, FRAME_DEADLINE_MS
from workload import FRAME_DATA_MB, PRESETS
from sim import CONGESTION

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "report")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- style
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STRAT = ["local", "edge", "v1", "v2"]
STRAT_NAME = {"local": "Always-local", "edge": "Always-edge", "v1": "Hybrid v1 (rule)", "v2": "Hybrid v2 (tree)"}
STRAT_COLOR = {"local": "#2a78d6", "edge": "#eb6834", "v1": "#1baf7a", "v2": "#eda100"}
NETS = ["good", "moderate", "measured", "poor"]
NET_LABEL = {"good": "Good\n4 ms", "moderate": "Moderate\n8 ms", "measured": "Measured\n22 ms", "poor": "Poor\n40 ms"}
COL_W = 3.5  # inches, one IEEE column

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 7.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.linewidth": 0.8,
})

rows = list(csv.DictReader(open(os.path.join(HERE, "results_calibrated.csv"), newline="")))


def avg(key, **flt):
    return float(np.mean([float(r[key]) for r in rows if all(r[k] == v for k, v in flt.items())]))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300)
    plt.close(fig)
    print("wrote", name)


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)


# ================================================================= Fig 1: system model
def box(ax, x, y, w, h, title, lines, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="white", edgecolor=edge, linewidth=1.1))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=7.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.52, "\n".join(lines), ha="center", va="top", fontsize=6.3,
            color=INK2, linespacing=1.35)


def arrow(ax, p, q, label=None, lpos=None, color=INK2, ha="center", style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, color=color, linewidth=0.9,
                                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text(*lpos, label, fontsize=6.1, color=INK2, ha=ha, va="center", linespacing=1.25)


fig, ax = plt.subplots(figsize=(COL_W, 2.2))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.3)
ax.axis("off")
# client boundary
ax.add_patch(FancyBboxPatch((0.08, 0.1), 5.0, 6.1, boxstyle="round,pad=0,rounding_size=0.15",
                            facecolor="#f4f4f1", edgecolor=BASE, linewidth=0.8, linestyle=(0, (3, 2))))
ax.text(0.3, 6.0, "Gaming PC (client)", fontsize=6.6, color=MUTED, va="top", fontweight="bold")
box(ax, 0.35, 3.55, 2.15, 2.0, "Game", ["one job per frame", "lognormal work $W_i$",
                                         "$D_i$ = 16.67 ms"], MUTED)
box(ax, 2.75, 3.55, 2.1, 2.0, "Local GPU", ["FIFO, delay $Q_L$", "speed $F_L$",
                                            "$K$ = 2 may wait"], STRAT_COLOR["local"])
box(ax, 0.35, 0.6, 4.5, 2.35, "Decision engine", ["estimates $T_L$ (1) and $T_E$ (2) from",
                                                   "$W_i$, $Q_L$, $F_L$, mean $L_{net}$, $B_{up}$, $B_{down}$,",
                                                   "and the last reported $Q_E$ (one job old)",
                                                   "v1: deadline rule   v2: decision tree"], STRAT_COLOR["v1"])
box(ax, 7.4, 1.5, 2.5, 2.9, "Edge server", ["shared GPU", "speed $F_E$ = 25", "queue delay $Q_E$",
                                             "cost: edge time"], STRAT_COLOR["edge"])
# network
ax.add_patch(FancyBboxPatch((5.45, 0.35), 1.6, 5.2, boxstyle="round,pad=0,rounding_size=0.3",
                            facecolor="#eef3fb", edgecolor="#86b6ef", linewidth=0.9))
ax.text(6.25, 5.38, "Network", ha="center", va="top", fontsize=7.5, fontweight="bold")
ax.text(6.25, 4.9, "RTT $L_{net}$\njitter $J$\nloss $p$", ha="center", va="top", fontsize=6.1,
        color=INK2, linespacing=1.35)
# arrows
arrow(ax, (1.425, 3.55), (1.425, 2.95))
ax.text(1.55, 3.25, "job $J_i$", fontsize=6.1, color=INK2, va="center")
arrow(ax, (3.8, 2.95), (3.8, 3.55))
ax.text(3.68, 3.25, "local", fontsize=6.1, color=INK2, va="center", ha="right")
arrow(ax, (4.85, 2.25), (7.4, 3.25))
ax.text(6.25, 3.28, "offload $D_{in}$", fontsize=6.1, color=INK2, ha="center", va="bottom")
arrow(ax, (7.4, 2.0), (4.85, 1.05))
ax.text(6.25, 1.2, "result $D_{out}$", fontsize=6.1, color=INK2, ha="center", va="top")
save(fig, "fig1_system_model.png")


# ================================================================= Fig 2: latency budget (Eq. 2)
def budget(net, work=PRESETS["moderate"], qe=CONGESTION["low"]):
    return dict(latency=net.latency_ms,
                upload=FRAME_DATA_MB / net.bandwidth_mbps * 1000,
                edge=qe + work / EDGE_SERVER.speed,
                download=0.4 / net.bandwidth_down_mbps * 1000)


nets_b = [("Good", budget(GOOD_NET)), ("Moderate", budget(OKAY_NET)),
          ("Measured", budget(MEASURED_NET)), ("Poor", budget(BAD_NET))]
comp = [("latency", "RTT $L_{net}$", "#2a78d6"), ("upload", "Upload $D_{in}/B_{up}$", "#eb6834"),
        ("edge", "$Q_E + W/F_E$", "#1baf7a"), ("download", "Download $D_{out}/B_{down}$", "#eda100")]

fig, ax = plt.subplots(figsize=(COL_W, 1.95))
fig.subplots_adjust(left=0.17, right=0.97, top=0.80, bottom=0.2)
for i, (name, parts) in enumerate(nets_b):
    left = 0
    for key, _, col in comp:
        ax.barh(i, parts[key], left=left, height=0.56, color=col, edgecolor="white", linewidth=0.8)
        left += parts[key]
    ax.text(max(left, FRAME_DEADLINE_MS) + 1.2, i, f"{left:.1f} ms", va="center", ha="left", fontsize=7, fontweight="bold")
ax.axvline(FRAME_DEADLINE_MS, color=INK, linewidth=1)
ax.text(FRAME_DEADLINE_MS + 0.8, 3.62, "deadline 16.67 ms", fontsize=6.5, va="center")
ax.set_yticks(range(len(nets_b)))
ax.set_yticklabels([n for n, _ in nets_b], fontsize=7.5, color=INK2)
ax.set_ylim(3.85, -0.5)
ax.set_xlim(0, 96)
ax.xaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="both", length=0)
ax.set_xlabel("Edge completion time $T_E$ (ms), 80-unit job, low congestion", fontsize=7)
fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in comp], loc="upper center",
           bbox_to_anchor=(0.54, 1.0), ncol=4, frameon=False, fontsize=6.3, handlelength=1.0,
           columnspacing=0.9, handletextpad=0.4)
save(fig, "fig2_latency_budget.png")

# ================================================================= Fig 3: miss rate, two stacked panels
BAR_W = 0.19
fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.9), sharex=True)
fig.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.11, hspace=0.32)
for ax, dev, ttl in zip(axes, ["current device", "aging device"],
                        ["(a) Current device ($F_L$ = 10)", "(b) Aging device ($F_L$ = 3.4)"]):
    x0 = np.arange(len(NETS))
    for j, s in enumerate(STRAT):
        vals = [avg("miss_rate_mean", strategy=s, device=dev, network=n) * 100 for n in NETS]
        xs = x0 + (j - 1.5) * (BAR_W + 0.015)
        ax.bar(xs, vals, width=BAR_W, color=STRAT_COLOR[s], zorder=3)
        for xv, v in zip(xs, vals):
            ax.text(xv, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.6, color=INK2, zorder=4)
    ax.set_title(ttl, loc="left", fontsize=7.5, fontweight="bold", pad=3)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Deadline-miss rate (%)", fontsize=7)
    grid_y(ax)
axes[1].set_xticks(np.arange(len(NETS)))
axes[1].set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=7)
fig.legend(handles=[Patch(facecolor=STRAT_COLOR[s], label=STRAT_NAME[s]) for s in STRAT], loc="upper center",
           bbox_to_anchor=(0.54, 1.0), ncol=4, frameon=False, fontsize=6.4, handlelength=1.0,
           columnspacing=0.9, handletextpad=0.4)
save(fig, "fig3_miss_rate.png")

# ================================================================= Fig 4: edge time (aging device)
fig, ax = plt.subplots(figsize=(COL_W, 1.95))
fig.subplots_adjust(left=0.12, right=0.985, top=0.84, bottom=0.22)
three = ["edge", "v1", "v2"]
x0 = np.arange(len(NETS))
for j, s in enumerate(three):
    vals = [avg("edge_ms_mean", strategy=s, device="aging device", network=n) / 1000 for n in NETS]
    xs = x0 + (j - 1) * (0.25 + 0.015)
    ax.bar(xs, vals, width=0.25, color=STRAT_COLOR[s], zorder=3)
    for xv, v in zip(xs, vals):
        ax.text(xv, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=5.6, color=INK2, zorder=4)
ax.set_xticks(x0)
ax.set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=6.8)
ax.set_ylabel("Edge time (s per 10 s)", fontsize=7)
ax.set_ylim(0, 2.3)
grid_y(ax)
fig.legend(handles=[Patch(facecolor=STRAT_COLOR[s], label=STRAT_NAME[s]) for s in three], loc="upper center",
           bbox_to_anchor=(0.54, 1.0), ncol=3, frameon=False, fontsize=6.4, handlelength=1.0,
           columnspacing=1.0, handletextpad=0.4)
save(fig, "fig4_edge_time_aging.png")
