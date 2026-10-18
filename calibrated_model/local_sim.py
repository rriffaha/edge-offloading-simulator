"""
local_sim.py — Step 2, Part B: one device running every job itself, over time.

Step 1 always used queue_ms = 0 (an idle device). Here the jobs arrive one
after another, so if the device is still busy when a new job arrives, that job
has to wait. This is the Q in the proposal's T_local = Q/f_L + w/f_L.
This file only answers: "if the device does everything locally, how late does
it get?"

Same units as model.py (work units, ms, Mb).
"""

import math
from dataclasses import dataclass

from model import Device, Job, NEW_DEVICE, AGING_DEVICE, t_local
from workload import PRESETS, SIGMA, generate_jobs


# A real GPU does not queue frames forever. The driver lets only a couple of
# frames wait (the render-ahead limit) and the game drops frames instead.
MAX_QUEUED_JOBS = 2


# ---------- What we record for each job ----------

@dataclass
class JobResult:
    name: str
    arrival_ms: float
    work: float
    queue_ms: float        # time spent waiting before the device started it
    completion_ms: float   # arrival -> finished (queue + compute)
    met_deadline: bool
    dropped: bool


# ---------- The simulation ----------

def simulate_local(jobs: list[Job], device: Device) -> list[JobResult]:
    """Run every job on the device, first in, first out."""
    results = []
    device_free_at = 0.0  # the time the device finishes everything it has accepted
    start_times = []      # start time of every accepted job, in order

    for job in jobs:
        arrival = job.arrival_ms

        # Jobs accepted earlier that have not started yet are still waiting.
        waiting = sum(1 for start in start_times if start > arrival)

        if waiting >= MAX_QUEUED_JOBS:
            results.append(JobResult(job.name, arrival, job.work,
                                     queue_ms=math.nan, completion_ms=math.nan,
                                     met_deadline=False, dropped=True))
            continue

        start = max(arrival, device_free_at)
        queue_ms = start - arrival
        completion = t_local(job, device, queue_ms)  # the Step 1 formula
        device_free_at = arrival + completion
        start_times.append(start)

        results.append(JobResult(job.name, arrival, job.work,
                                 queue_ms=queue_ms, completion_ms=completion,
                                 met_deadline=completion <= job.deadline_ms,
                                 dropped=False))
    return results


# ---------- Turning the per-job records into four numbers ----------

@dataclass
class Summary:
    avg_completion_ms: float  # over the jobs that actually ran
    miss_rate: float          # late + dropped, as a fraction of all jobs
    drop_rate: float          # dropped, as a fraction of all jobs
    max_queue_ms: float


def summarize(results: list[JobResult]) -> Summary:
    ran = [r for r in results if not r.dropped]
    missed = [r for r in results if not r.met_deadline]  # dropped jobs count here too
    return Summary(
        avg_completion_ms=sum(r.completion_ms for r in ran) / len(ran),
        miss_rate=len(missed) / len(results),
        drop_rate=(len(results) - len(ran)) / len(results),
        max_queue_ms=max(r.queue_ms for r in ran),
    )


# ---------- Quick demo ----------

if __name__ == "__main__":
    N_FRAMES = 600  # 10 seconds at 60 fps
    SEED = 1

    print(f"{N_FRAMES} frames, sigma = {SIGMA}, seed = {SEED}, MAX_QUEUED_JOBS = {MAX_QUEUED_JOBS}")
    print(f"{'device':<13} {'preset':<9} {'avg done ms':>11} {'miss %':>7} {'drop %':>7} {'max queue ms':>13}")
    print("-" * 65)
    for device in [NEW_DEVICE, AGING_DEVICE]:
        for name, mean_work in PRESETS.items():
            s = summarize(simulate_local(generate_jobs(N_FRAMES, mean_work, SIGMA, SEED), device))
            print(f"{device.name:<13} {name:<9} {s.avg_completion_ms:>11.1f} {s.miss_rate:>7.1%} {s.drop_rate:>7.1%} {s.max_queue_ms:>13.1f}")

    print()
    print("first 15 jobs, aging device + moderate")
    print(f"{'job':<9} {'arrival':>8} {'work':>7} {'queue_ms':>9} {'done_ms':>8}  {'on time?':<9} dropped?")
    print("-" * 62)
    jobs = generate_jobs(N_FRAMES, PRESETS["moderate"], SIGMA, SEED)
    for r in simulate_local(jobs, AGING_DEVICE)[:15]:
        if r.dropped:
            print(f"{r.name:<9} {r.arrival_ms:>8.1f} {r.work:>7.1f} {'-':>9} {'-':>8}  {'NO':<9} DROPPED")
        else:
            on_time = "yes" if r.met_deadline else "NO"
            print(f"{r.name:<9} {r.arrival_ms:>8.1f} {r.work:>7.1f} {r.queue_ms:>9.1f} {r.completion_ms:>8.1f}  {on_time:<9} no")

    print()
    print("sanity check 1: sigma = 0 on the new device (every job identical)")
    for name, mean_work in PRESETS.items():
        jobs = generate_jobs(N_FRAMES, mean_work, 0.0, SEED)
        results = simulate_local(jobs, NEW_DEVICE)
        queue_zero = all(r.queue_ms == 0 for r in results)
        matches_step1 = all(r.completion_ms == t_local(job, NEW_DEVICE) for r, job in zip(results, jobs))
        print(f"{name:<9} work = {jobs[0].work:>5.1f}  work / speed = {mean_work / NEW_DEVICE.speed:>4.1f} ms  "
              f"completion = {results[0].completion_ms:>4.1f} ms  queue always 0? {queue_zero}  "
              f"all {N_FRAMES} match Step 1? {matches_step1}")

    print()
    print("sanity check 2: same seed -> identical summary")
    first = summarize(simulate_local(generate_jobs(N_FRAMES, PRESETS["moderate"], SIGMA, SEED), AGING_DEVICE))
    second = summarize(simulate_local(generate_jobs(N_FRAMES, PRESETS["moderate"], SIGMA, SEED), AGING_DEVICE))
    other = summarize(simulate_local(generate_jobs(N_FRAMES, PRESETS["moderate"], SIGMA, SEED + 1), AGING_DEVICE))
    print(f"run 1 (seed {SEED}): {first}")
    print(f"run 2 (seed {SEED}): {second}")
    print(f"identical? {first == second}")
    print(f"seed {SEED + 1} gives a different summary? {first != other}")
