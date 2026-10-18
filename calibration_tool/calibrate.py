"""
calibrate.py - measure REAL numbers on this laptop to replace the placeholders in model.py.
Standard library + numpy only (PyTorch is optional, used for the GPU test).

    python calibrate.py net                       # latency, jitter, loss, up/down bandwidth
    python calibrate.py compute                   # CPU speed (and GPU speed with PyTorch: NVIDIA CUDA or Apple MPS)
    python calibrate.py frames path\\to\\log.csv    # real game frame times (PresentMon / CapFrameX CSV)
    python calibrate.py all [log.csv]             # everything

net / compute / all save measurements/calibration_<hostname>.json (overwritten each run).
Run it with the laptop plugged in, in Windows "Best performance" mode, nothing heavy open.
See README.md in this folder for setup.
"""
import csv
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
import urllib.request

DEADLINE_MS = 1000 / 60
UA = {"User-Agent": "Mozilla/5.0 (calibrate.py)"}


# ---------------------------------------------------------------- network
# Maps to Network(latency_ms, bandwidth_mbps, jitter_ms, loss_rate, bandwidth_down_mbps)

def tcp_rtts(host, port=443, n=100, timeout=1.0, gap=0.05):
    """Time n TCP connects. One connect is about one round trip."""
    rtts, lost = [], 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                rtts.append((time.perf_counter() - t0) * 1000)
        except OSError:
            lost += 1
        time.sleep(gap)
    return rtts, lost


def measure_latency(hosts=("1.1.1.1", "8.8.8.8")):
    out = {}
    for host in hosts:
        print(f"  pinging {host} ...")
        rtts, lost = tcp_rtts(host)
        if len(rtts) >= 2:
            out[host] = dict(latency_ms=statistics.median(rtts),
                             jitter_ms=statistics.stdev(rtts),
                             loss_rate=lost / (len(rtts) + lost),
                             samples=len(rtts))
        else:
            out[host] = dict(error="no replies")
    return out


def download_mbps(mb=25):
    url = f"https://speed.cloudflare.com/__down?bytes={mb * 1_000_000}"
    t0, n = time.perf_counter(), 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        while chunk := r.read(1 << 16):
            n += len(chunk)
    return n * 8 / 1e6 / (time.perf_counter() - t0)


def upload_mbps(mb=10):
    data = bytes(mb * 1_000_000)
    req = urllib.request.Request("https://speed.cloudflare.com/__up", data=data, method="POST",
                                 headers={**UA, "Content-Type": "application/octet-stream"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    return len(data) * 8 / 1e6 / (time.perf_counter() - t0)


def measure_network():
    res = {"latency": measure_latency()}
    for name, fn in (("download_mbps", download_mbps), ("upload_mbps", upload_mbps)):
        try:
            print(f"  measuring {name} (3 runs) ...")
            res[name] = statistics.median(fn() for _ in range(3))
        except Exception as e:  # no internet, blocked, etc.
            res[name] = None
            res[name + "_error"] = str(e)
    return res


# ---------------------------------------------------------------- compute
# Defines ONE "work unit" = one 2048x2048 float32 matrix multiply (about 17 GFLOP).
# The absolute number is arbitrary; what matters is the RATIO between devices.

def bench(fn, sync=lambda: None, warm_s=2.0, iters=30):
    t_end = time.perf_counter() + warm_s      # warm up by TIME so GPU clocks ramp up
    while time.perf_counter() < t_end:
        fn()
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def measure_compute(n=2048):
    import numpy as np
    out = {}
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    ms = bench(lambda: a @ b, warm_s=0.5, iters=10)
    out["cpu"] = dict(ms_per_unit=ms, units_per_ms=1 / ms, tflops=2 * n**3 / ms / 1e9)
    try:
        import torch
        mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if torch.cuda.is_available():
            dev, sync, name = "cuda", torch.cuda.synchronize, torch.cuda.get_device_name(0)
        elif mps_ok:                                   # Apple-silicon Mac
            dev, sync, name = "mps", torch.mps.synchronize, "Apple GPU (MPS)"
        else:
            dev = None
        if dev:
            x = torch.rand(n, n, device=dev)
            y = torch.rand(n, n, device=dev)
            ms = bench(lambda: x @ y, sync, warm_s=2.0, iters=100)
            out["gpu"] = dict(name=name, ms_per_unit=ms, units_per_ms=1 / ms, tflops=2 * n**3 / ms / 1e9)
        else:
            out["gpu"] = "PyTorch found no NVIDIA (CUDA) or Apple (MPS) GPU - GPU not measured"
    except ImportError:
        out["gpu"] = "PyTorch not installed - GPU not measured (pip install torch)"
    return out


# ---------------------------------------------------------------- real frame times
# Gives the real frame-time distribution -> replaces SIGMA and the workload means.

def analyze_frames(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = {c.lower(): c for c in rd.fieldnames}
        key = next((cols[k] for k in ("msbetweenpresents", "frametime", "frametime(ms)") if k in cols), None)
        if key is None:
            raise SystemExit(f"No frame-time column found. Columns: {rd.fieldnames}")
        ft = []
        for r in rd:
            try:
                ft.append(float(r[key]))
            except (ValueError, TypeError):
                pass
    ft = [x for x in ft if x > 0]
    q = statistics.quantiles(ft, n=100)
    return dict(
        frames=len(ft),
        mean_ms=statistics.mean(ft),
        p50_ms=q[49], p95_ms=q[94], p99_ms=q[98],
        pct_over_16_67ms=100 * sum(x > DEADLINE_MS for x in ft) / len(ft),
        lognormal_sigma=statistics.stdev(math.log(x) for x in ft),
    )


# ---------------------------------------------------------------- main
def show(title, data):
    print(f"\n== {title} ==")
    print(json.dumps(data, indent=2))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("net", "compute", "frames", "all"):
        raise SystemExit(__doc__)
    what = sys.argv[1]
    result = {"host": platform.node(), "platform": platform.platform(), "processor": platform.processor()}

    if what == "frames" or (what == "all" and len(sys.argv) > 2):
        path = sys.argv[2]
        result["frames"] = analyze_frames(path)
        show("frame times", result["frames"])
    if what in ("compute", "all"):
        print("Measuring compute speed ...")
        result["compute"] = measure_compute()
        show("compute (1 unit = one 2048x2048 fp32 matmul)", result["compute"])
    if what in ("net", "all"):
        print("Measuring network ...")
        result["network"] = measure_network()
        show("network", result["network"])
        ok = {h: v for h, v in result["network"]["latency"].items() if "latency_ms" in v}
        if ok and result["network"].get("upload_mbps"):
            h = min(ok, key=lambda k: ok[k]["latency_ms"])
            v = ok[h]
            print("\nPaste into model.py as a measured network:")
            print(f'MEASURED_NET = Network("measured", latency_ms={v["latency_ms"]:.1f}, '
                  f'bandwidth_mbps={result["network"]["upload_mbps"]:.0f}, jitter_ms={v["jitter_ms"]:.1f}, '
                  f'loss_rate={v["loss_rate"]:.3f}, bandwidth_down_mbps={result["network"]["download_mbps"]:.0f})')

    if what != "frames":
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurements")
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.join(out_dir, f"calibration_{platform.node()}.json")
        with open(fn, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved {fn}")


if __name__ == "__main__":
    main()
