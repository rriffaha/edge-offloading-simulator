"""
model.py — the core formulas of the working model.

One "job" = a chunk of rendering work the game needs done.
The device can run it itself (local) or send it to a nearby edge server (offload).
This file only answers: "how long would this job take each way?"

Units used everywhere:
    work        -> work units (an abstract amount of compute)
    speed       -> work units per millisecond
    data        -> megabits (Mb)
    bandwidth   -> megabits per second (Mbps)
    time        -> milliseconds (ms)
"""

from dataclasses import dataclass


# ---------- The three things we describe ----------

@dataclass
class Device:
    name: str
    speed: float  # work units per ms


@dataclass
class Network:
    name: str
    latency_ms: float      # round-trip delay to the edge server
    bandwidth_mbps: float  # uplink: how fast data can be sent to the edge (B_up)
    jitter_ms: float = 0.0  # std dev of the latency from job to job (used from Step 3 on)
    loss_rate: float = 0.0  # chance an offloaded job is lost in transit (used from Step 3 on)
    bandwidth_down_mbps: float = 0.0  # downlink (B_down); 0 means "same as the uplink"

    def __post_init__(self):
        if self.bandwidth_down_mbps == 0.0:
            self.bandwidth_down_mbps = self.bandwidth_mbps


@dataclass
class Job:
    name: str
    work: float      # work units needed
    data_mb: float   # megabits that must be sent if offloaded (D_in)
    deadline_ms: float
    arrival_ms: float = 0.0  # when the job shows up (used from Step 2 on)
    data_out_mb: float = 0.4  # megabits the edge sends back (D_out, used from Step 3 on)


# ---------- Parameters (MEASURED where marked, otherwise assumed) ----------
# Only speed RATIOS matter: the work unit is abstract, so the current device is
# fixed at 10 as the reference and the other speeds are scaled against it.

NEW_DEVICE   = Device("current device", speed=10.0)  # reference: RTX 5070 Ti laptop (Zephyrus)
AGING_DEVICE = Device("aging device", speed=3.4)     # MEASURED: GTX 1660 Ti = 34% of RTX 5070 Ti (3 runs each, same calibrate.py)
EDGE_SERVER  = Device("edge server",  speed=25.0)    # ASSUMED: 2.5x the reference (not measurable without an edge server)

FRAME_DEADLINE_MS = 1000 / 60  # one frame at 60 fps = 16.67 ms

# good / moderate / poor are ASSUMED edge scenarios (a nearby server at different quality levels)
GOOD_NET = Network("good",     latency_ms=4,  bandwidth_mbps=100, jitter_ms=1,  loss_rate=0.005, bandwidth_down_mbps=200)
OKAY_NET = Network("moderate", latency_ms=8,  bandwidth_mbps=50,  jitter_ms=3,  loss_rate=0.01,  bandwidth_down_mbps=100)
# MEASURED (calibrate.py, 2 laptops, 4 runs, medians): latency 17-28 ms, upload 39-50, download 88-251 Mbps
MEASURED_NET = Network("measured", latency_ms=22, bandwidth_mbps=44, jitter_ms=7, loss_rate=0.003, bandwidth_down_mbps=150)
BAD_NET  = Network("poor",     latency_ms=40, bandwidth_mbps=10,  jitter_ms=10, loss_rate=0.05,  bandwidth_down_mbps=20)

# Two fixed jobs for the demo below only; the experiments use workload.py instead.
LIGHT_JOB = Job("light", work=80,  data_mb=0.2, deadline_ms=FRAME_DEADLINE_MS)
HEAVY_JOB = Job("heavy", work=220, data_mb=0.2, deadline_ms=FRAME_DEADLINE_MS)


# ---------- The two formulas ----------

def t_local(job: Job, device: Device, queue_ms: float = 0.0) -> float:
    """T_local = queue + work / device speed"""
    return queue_ms + job.work / device.speed


def t_edge(job: Job, network: Network, edge: Device = EDGE_SERVER, qe_ms: float = 0.0) -> float:
    """T_edge = L_net + D_in/B_up + Q_E + W/F_E + D_out/B_down   (Eq. 2 of the proposal)

    qe_ms is the edge queueing delay: how long the job waits at the shared server
    before it is processed (0 = idle server).
    """
    upload_ms = job.data_mb / network.bandwidth_mbps * 1000            # Mb / Mbps -> s -> ms
    download_ms = job.data_out_mb / network.bandwidth_down_mbps * 1000
    return network.latency_ms + upload_ms + qe_ms + job.work / edge.speed + download_ms


# ---------- Quick demo: run every combination and print a table ----------

if __name__ == "__main__":
    print(f"deadline = {FRAME_DEADLINE_MS:.1f} ms")
    print(f"{'device':<13} {'job':<6} {'network':<8} {'T_local':>8} {'T_edge':>8}  {'faster':<7} {'local ok?':<10} edge ok?")
    print("-" * 78)
    for device in [NEW_DEVICE, AGING_DEVICE]:
        for job in [LIGHT_JOB, HEAVY_JOB]:
            for net in [GOOD_NET, OKAY_NET, MEASURED_NET, BAD_NET]:
                tl = t_local(job, device)
                te = t_edge(job, net)
                faster = "local" if tl <= te else "edge"
                local_ok = "yes" if tl <= job.deadline_ms else "NO"
                edge_ok = "yes" if te <= job.deadline_ms else "NO"
                print(f"{device.name:<13} {job.name:<6} {net.name:<8} {tl:>8.1f} {te:>8.1f}  {faster:<7} {local_ok:<10} {edge_ok}")
