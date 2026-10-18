"""
run_experiments.py — the pre-calibration baseline (placeholder parameters).

Runs the proposal's evaluation grid: 2 devices x 3 workloads x 3 networks x
2 edge congestion levels x 4 strategies, over 20 fixed seeds. Writes every
row to results.csv (next to this file) and prints the headline tables.
Kept for comparison with ../calibrated_model; see the README.
"""

import csv
import os
from dataclasses import fields

import numpy as np

from model import NEW_DEVICE, AGING_DEVICE, GOOD_NET, OKAY_NET, BAD_NET
from workload import PRESETS, SIGMA, generate_jobs
from sim import CONGESTION, Metrics, make_tree_policy, policy_edge, policy_local, policy_v1, simulate, train_tree

N_FRAMES = 600
SEEDS = range(1, 21)
DEVICES = [NEW_DEVICE, AGING_DEVICE]
NETWORKS = [GOOD_NET, OKAY_NET, BAD_NET]
FIELDS = [f.name for f in fields(Metrics)]
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")


if __name__ == "__main__":
    clf, test_acc, offload_share = train_tree()
    print(f"decision tree (v2): depth {clf.get_depth()}, {clf.get_n_leaves()} leaves, "
          f"held-out accuracy {test_acc:.1%}, {offload_share:.0%} of training labels = edge")

    strategies = {"local": policy_local, "edge": policy_edge, "v1": policy_v1, "v2": make_tree_policy(clf)}
    job_cache = {(p, s): generate_jobs(N_FRAMES, PRESETS[p], SIGMA, s) for p in PRESETS for s in SEEDS}

    rows = []
    for device in DEVICES:
        for preset in PRESETS:
            for net in NETWORKS:
                for cong, qe_mean in CONGESTION.items():
                    for name, policy in strategies.items():
                        runs = [simulate(job_cache[(preset, s)], device, net, qe_mean, policy, s) for s in SEEDS]
                        row = {"device": device.name, "workload": preset, "network": net.name,
                               "congestion": cong, "strategy": name}
                        for f in FIELDS:
                            vals = [getattr(m, f) for m in runs]
                            row[f + "_mean"], row[f + "_std"] = float(np.nanmean(vals)), float(np.nanstd(vals))
                        rows.append(row)

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def avg(sel, field):
        return np.mean([r[field + "_mean"] for r in rows if sel(r)])

    print(f"\n{N_FRAMES} frames, {len(SEEDS)} seeds, {len(rows)} rows -> results.csv\n")

    print("Overall, averaged over every scenario")
    print(f"{'strategy':<9} {'miss %':>7} {'violation ms':>13} {'edge s':>7} {'offloaded %':>12} {'data Mb':>8}")
    for s in strategies:
        sel = lambda r, s=s: r["strategy"] == s
        print(f"{s:<9} {avg(sel, 'miss_rate'):>7.1%} {avg(sel, 'avg_violation_ms'):>13.2f} "
              f"{avg(sel, 'edge_ms') / 1000:>7.2f} {avg(sel, 'pct_offloaded'):>12.1%} {avg(sel, 'data_sent_mb'):>8.0f}")

    for title, key, values in [("By network", "network", [n.name for n in NETWORKS]),
                               ("By edge congestion", "congestion", list(CONGESTION)),
                               ("By device", "device", [d.name for d in DEVICES]),
                               ("By workload", "workload", list(PRESETS))]:
        print(f"\n{title}: miss % | edge s")
        print(f"{key:<15} " + " ".join(f"{s:>13}" for s in strategies))
        for v in values:
            cells = []
            for s in strategies:
                sel = lambda r, s=s, v=v: r["strategy"] == s and r[key] == v
                cells.append(f"{avg(sel, 'miss_rate'):>6.1%} |{avg(sel, 'edge_ms') / 1000:>5.2f}")
            print(f"{v:<15} " + " ".join(f"{c:>13}" for c in cells))
