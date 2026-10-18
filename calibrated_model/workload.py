"""
workload.py — Step 2, Part A: a stream of jobs over time.

Step 1 looked at one job at a time. A real game produces one job per frame,
60 times a second, and the jobs are not all the same size.
This file only answers: "what jobs arrive, when, and how big is each one?"

Same units as model.py (work units, ms, Mb).
"""

import math

import numpy as np

from model import Job, FRAME_DEADLINE_MS


# ---------- Workload presets (ASSUMED; to be replaced by real game frame times, see calibrate.py frames) ----------

SIGMA = 0.3          # spread of the work per job (0 = every job identical)
FRAME_DATA_MB = 0.2  # same data size as the Step 1 jobs

# name -> mean work per job
PRESETS = {
    "light":    50,
    "moderate": 80,
    "heavy":    120,
}


# ---------- The generator ----------

def generate_jobs(n_frames: int, mean_work: float, sigma: float, seed: int) -> list[Job]:
    """One job per frame, arriving every FRAME_DEADLINE_MS, with lognormal work."""
    rng = np.random.default_rng(seed)

    # numpy's lognormal takes the mean of the *underlying normal*, not the mean
    # we want. This shift makes the average work come out as mean_work.
    mu = math.log(mean_work) - sigma**2 / 2
    works = rng.lognormal(mean=mu, sigma=sigma, size=n_frames)

    jobs = []
    for i in range(n_frames):
        jobs.append(Job(
            name=f"frame {i}",
            work=float(works[i]),
            data_mb=FRAME_DATA_MB,
            deadline_ms=FRAME_DEADLINE_MS,
            arrival_ms=i * FRAME_DEADLINE_MS,
        ))
    return jobs


# ---------- Quick demo ----------

if __name__ == "__main__":
    N_FRAMES = 600  # 10 seconds at 60 fps
    SEED = 1

    jobs = generate_jobs(N_FRAMES, PRESETS["moderate"], SIGMA, SEED)

    print(f"first 10 jobs (moderate, sigma = {SIGMA}, seed = {SEED})")
    print(f"{'job':<9} {'arrival_ms':>10} {'work':>8} {'deadline_ms':>12}")
    print("-" * 42)
    for job in jobs[:10]:
        print(f"{job.name:<9} {job.arrival_ms:>10.2f} {job.work:>8.1f} {job.deadline_ms:>12.2f}")

    print()
    print(f"work over {N_FRAMES} frames (seed = {SEED})")
    print(f"{'preset':<9} {'target':>7} {'mean':>8} {'min':>8} {'max':>8}")
    print("-" * 44)
    for name, mean_work in PRESETS.items():
        works = [job.work for job in generate_jobs(N_FRAMES, mean_work, SIGMA, SEED)]
        print(f"{name:<9} {mean_work:>7} {np.mean(works):>8.1f} {min(works):>8.1f} {max(works):>8.1f}")

    print()
    print("reproducibility (first 3 work values, moderate)")
    for label, seed in [("seed 1", 1), ("seed 1 again", 1), ("seed 2", 2)]:
        works = [job.work for job in generate_jobs(N_FRAMES, PRESETS["moderate"], SIGMA, seed)]
        print(f"{label:<13} {works[0]:>7.2f} {works[1]:>7.2f} {works[2]:>7.2f}")

    same_seed = generate_jobs(N_FRAMES, PRESETS["moderate"], SIGMA, 1) == generate_jobs(N_FRAMES, PRESETS["moderate"], SIGMA, 1)
    diff_seed = generate_jobs(N_FRAMES, PRESETS["moderate"], SIGMA, 1) == generate_jobs(N_FRAMES, PRESETS["moderate"], SIGMA, 2)
    print(f"seed 1 vs seed 1: all {N_FRAMES} jobs identical? {same_seed}")
    print(f"seed 1 vs seed 2: all {N_FRAMES} jobs identical? {diff_seed}")
