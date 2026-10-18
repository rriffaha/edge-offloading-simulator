"""
make_figures.py - report figures from the calibrated simulation results.

Run from this folder:   python make_figures.py
Needs: pip install matplotlib numpy
Reads:  results_calibrated.csv and results_placeholder.csv (both in this folder)
Writes: figures\\*.png and figures\\summary_tables.csv
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- style
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
# categorical slots 1-4 (validated): one fixed colour per strategy, everywhere
STRAT = ["local", "edge", "v1", "v2"]
STRAT_NAME = {"local": "Always-local", "edge": "Always-edge",
              "v1": "Hybrid v1 (rule)", "v2": "Hybrid v2 (decision tree)"}
STRAT_COLOR = {"local": "#2a78d6", "edge": "#eb6834", "v1": "#1baf7a", "v2": "#eda100"}
NETS = ["good", "moderate", "poor", "measured"]
NET_LABEL = {"good": "Good\n4 ms RTT", "moderate": "Moderate\n8 ms RTT",
             "poor": "Poor\n40 ms RTT", "measured": "Measured\n22 ms RTT"}
DEADLINE = 1000 / 60

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.linewidth": 1,
})


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        path = os.path.join(HERE, "..", name)
    return list(csv.DictReader(open(path, newline="")))


new = load("results_calibrated.csv")
old = load("results_placeholder.csv")


def avg(rows, key, **flt):
    vals = [float(r[key]) for r in rows if all(r[k] == v for k, v in flt.items())]
    return float(np.mean(vals))


def style_axes(ax, ymax=None):
    ax.yaxis.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.tick_params(axis="x", labelcolor=INK2, labelsize=9)
    if ymax:
        ax.set_ylim(0, ymax)


def headline(fig, title, subtitle):
    fig.text(0.015, 0.975, title, fontsize=14, fontweight="semibold", color=INK, va="top", ha="left")
    fig.text(0.015, 0.925, subtitle, fontsize=9.5, color=INK2, va="top", ha="left")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote", name)


BAR_W = 0.17  # thin bars, small gap between neighbours (the surface shows through)

# ================================================================= Fig 1: latency budget
# Why offloading fails: add up Eq. 2 for a moderate job (work 80) on each network.
def budget(lat, up_mbps, down_mbps, qe=1.0, work=80, edge_speed=25, d_in=0.2, d_out=0.4):
    return dict(latency=lat, upload=d_in / up_mbps * 1000, edge=qe + work / edge_speed,
                download=d_out / down_mbps * 1000)

nets_b = [("Good", budget(4, 100, 200)), ("Moderate", budget(8, 50, 100)),
          ("Measured", budget(22, 44, 150)), ("Poor", budget(40, 10, 20))]
comp = [("latency", "Network delay (L_net)", "#2a78d6"), ("upload", "Upload (D_in / B_up)", "#eb6834"),
        ("edge", "Edge wait + processing (Q_E + W/F_E)", "#1baf7a"),
        ("download", "Download (D_out / B_down)", "#eda100")]

fig, ax = plt.subplots(figsize=(10, 3.9))
fig.subplots_adjust(left=0.10, right=0.97, top=0.70, bottom=0.14)
for i, (name, parts) in enumerate(nets_b):
    left = 0
    for key, _, col in comp:
        v = parts[key]
        ax.barh(i, v, left=left, height=0.5, color=col, edgecolor=SURFACE, linewidth=2)
        left += v
    ax.text(max(left, DEADLINE) + 1.4, i, f"{left:.0f} ms", va="center", ha="left", fontsize=10,
            fontweight="semibold", color=INK)
ax.axvline(DEADLINE, color=INK, linewidth=1.5)
ax.text(DEADLINE + 0.8, -0.62, "Frame deadline 16.67 ms", fontsize=9, color=INK, va="center", ha="left")
ax.set_yticks(range(len(nets_b)))
ax.set_yticklabels([n for n, _ in nets_b], fontsize=10.5, color=INK2)
ax.invert_yaxis()
ax.set_xlim(0, 95)
ax.set_ylim(3.6, -0.95)
ax.xaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.tick_params(axis="both", length=0, labelsize=9)
ax.set_xlabel("Round-trip time to finish one offloaded job (ms), moderate workload, low edge congestion", fontsize=9)
headline(fig, "Offloading a frame only fits the deadline on a nearby edge",
         "Eq. 2 summed for an 80-unit job. The measured home-internet path needs about 33 ms, twice the 16.67 ms frame budget.")
fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in comp], loc="upper left",
           bbox_to_anchor=(0.10, 0.865), ncol=4, frameon=False, fontsize=8.5, handlelength=1.2,
           columnspacing=1.4, labelcolor=INK2)
save(fig, "fig1_latency_budget.png")

# ================================================================= Fig 2: miss rate
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.14, wspace=0.06)
for ax, dev, ttl in zip(axes, ["current device", "aging device"],
                        ["Current device (speed 10)", "Aging device (speed 3.4)"]):
    x0 = np.arange(len(NETS))
    for j, s in enumerate(STRAT):
        vals = [avg(new, "miss_rate_mean", strategy=s, device=dev, network=n) * 100 for n in NETS]
        xs = x0 + (j - 1.5) * (BAR_W + 0.02)
        ax.bar(xs, vals, width=BAR_W, color=STRAT_COLOR[s], zorder=3)
        for xv, v in zip(xs, vals):
            ax.text(xv, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2, zorder=4)
    ax.set_xticks(x0)
    ax.set_xticklabels([NET_LABEL[n] for n in NETS])
    ax.set_title(ttl, loc="left", fontsize=10.5, color=INK, pad=8, fontweight="semibold")
    style_axes(ax, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
axes[0].set_ylabel("Deadline-miss rate (%)", fontsize=9)
axes[1].tick_params(axis="y", labelleft=False)
headline(fig, "Hybrids cut deadline misses wherever offloading can help",
         "Share of frames that finish late, are dropped, or are lost. Mean over 3 workloads x 2 congestion levels x 20 seeds.")
fig.legend(handles=[Patch(facecolor=STRAT_COLOR[s], label=STRAT_NAME[s]) for s in STRAT], loc="upper left",
           bbox_to_anchor=(0.06, 0.875), ncol=4, frameon=False, fontsize=9, handlelength=1.2,
           columnspacing=1.6, labelcolor=INK2)
save(fig, "fig2_miss_rate.png")

# ================================================================= Fig 3: edge time (cost)
fig, ax = plt.subplots(figsize=(10, 4.4))
fig.subplots_adjust(left=0.07, right=0.985, top=0.74, bottom=0.17)
three = ["edge", "v1", "v2"]
x0 = np.arange(len(NETS))
for j, s in enumerate(three):
    vals = [avg(new, "edge_ms_mean", strategy=s, network=n) / 1000 for n in NETS]
    xs = x0 + (j - 1) * (BAR_W + 0.02)
    ax.bar(xs, vals, width=BAR_W, color=STRAT_COLOR[s], zorder=3)
    for xv, v in zip(xs, vals):
        ax.text(xv, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2, zorder=4)
ax.set_xticks(x0)
ax.set_xticklabels([NET_LABEL[n] for n in NETS])
style_axes(ax, 2.35)
ax.set_ylabel("Edge server time used (s per 10 s of play)", fontsize=9)
headline(fig, "The hybrids use a fraction of the edge time of always-edge",
         "Edge processing time, the provider-cost proxy. Always-local uses none. v2 uses the least and none on the poor network.")
fig.legend(handles=[Patch(facecolor=STRAT_COLOR[s], label=STRAT_NAME[s]) for s in three], loc="upper left",
           bbox_to_anchor=(0.07, 0.865), ncol=3, frameon=False, fontsize=9, handlelength=1.2,
           columnspacing=1.6, labelcolor=INK2)
save(fig, "fig3_edge_time.png")

# ================================================================= Fig 4: calibration effect
# same 3 placeholder networks, aging device: placeholder speed 4.0 (old run) vs measured 3.4 (new run)
nets3 = {"good", "moderate", "poor"}
new3 = [r for r in new if r["network"] in nets3]
rows = []
for s in STRAT:
    o = avg(old, "miss_rate_mean", strategy=s, device="aging device") * 100
    n = avg(new3, "miss_rate_mean", strategy=s, device="aging device") * 100
    rows.append((s, o, n))
BEFORE, AFTER = "#86b6ef", "#1c5cab"   # blue ramp steps 250 / 550: one hue, two shades
fig, ax = plt.subplots(figsize=(10, 3.7))
fig.subplots_adjust(left=0.23, right=0.97, top=0.70, bottom=0.17)
for i, (s, o, n) in enumerate(rows):
    ax.plot([o, n], [i, i], color=BASE, linewidth=2, solid_capstyle="round", zorder=2)
    ax.scatter([o], [i], s=110, color=BEFORE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.scatter([n], [i], s=110, color=AFTER, edgecolor=SURFACE, linewidth=2, zorder=4)
    if abs(n - o) < 0.5:
        ax.text(n + 2.2, i, f"{n:.1f}  (no change)", va="center", fontsize=9, color=INK2)
    else:
        ax.text(o - 2.2, i, f"{o:.1f}", va="center", ha="right", fontsize=9, color=INK2)
        ax.text(n + 2.2, i, f"{n:.1f}", va="center", ha="left", fontsize=9, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([STRAT_NAME[s] for s, _, _ in rows], fontsize=10, color=INK2)
ax.invert_yaxis()
ax.set_xlim(35, 100)
ax.xaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.tick_params(axis="both", length=0, labelsize=9)
ax.set_xlabel("Deadline-miss rate on the aging device (%), good / moderate / poor networks", fontsize=9)
headline(fig, "Measuring the aging device made it slower than assumed",
         "Aging speed 4.0 (placeholder) vs 3.4 (measured GTX 1660 Ti). Strategy ranking is unchanged.")
fig.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=BEFORE, markeredgecolor="none", markersize=9, label="Placeholder (speed 4.0)"),
                    Line2D([0], [0], marker="o", color="none", markerfacecolor=AFTER, markeredgecolor="none", markersize=9, label="Measured (speed 3.4)")],
           loc="upper left", bbox_to_anchor=(0.23, 0.865), ncol=2, frameon=False, fontsize=9,
           labelcolor=INK2, handletextpad=0.3, columnspacing=1.8)
save(fig, "fig4_calibration_effect.png")

# ================================================================= summary table (CSV)
with open(os.path.join(OUT, "summary_tables.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "value", "strategy", "miss_rate_pct", "avg_violation_ms", "edge_time_s", "offloaded_pct"])
    groups = [("overall", [None])] + [("network", NETS), ("device", ["current device", "aging device"]),
                                       ("workload", ["light", "moderate", "heavy"]), ("congestion", ["low", "high"])]
    for gname, vals in groups:
        for v in vals:
            flt = {} if v is None else {gname: v}
            for s in STRAT:
                w.writerow([gname, v or "all", STRAT_NAME[s],
                            f"{avg(new, 'miss_rate_mean', strategy=s, **flt) * 100:.1f}",
                            f"{avg(new, 'avg_violation_ms_mean', strategy=s, **flt):.2f}",
                            f"{avg(new, 'edge_ms_mean', strategy=s, **flt) / 1000:.2f}",
                            f"{avg(new, 'pct_offloaded_mean', strategy=s, **flt) * 100:.1f}"])
print("wrote summary_tables.csv")
