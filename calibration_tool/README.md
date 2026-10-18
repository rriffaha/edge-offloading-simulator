# calibrate.py: measure a laptop (about 10 minutes)

Replaces assumed simulator inputs with real measurements. It runs a matrix-multiply benchmark on the
CPU and GPU, and times connections and transfers to Cloudflare and Google. It does not collect or
send any personal data.

| Command | Measures | Used for in `model.py` |
|---|---|---|
| `python calibrate.py compute` | ms per work unit on CPU and GPU (1 unit = one 2048x2048 fp32 matrix multiply) | device speed ratio |
| `python calibrate.py net` | latency (median TCP connect time), jitter (its std), loss (failed connects), upload and download speed | `MEASURED_NET` |
| `python calibrate.py frames log.csv` | frame-time percentiles and lognormal sigma from a PresentMon / CapFrameX log | workload sizes and `SIGMA` |
| `python calibrate.py all [log.csv]` | all of the above | |

`compute`, `net` and `all` save `measurements/calibration_<computer-name>.json`, overwritten on
each run, so keep the printed output of every run as well.

## Setup (once)

1. Python 3.10 or newer (`python --version`; on a Mac use `python3`).
2. `python -m pip install numpy`
3. GPU benchmark (optional, but the GPU decides the device speed):
   - NVIDIA GPU: `python -m pip install torch --index-url https://download.pytorch.org/whl/cu128`
     (a few GB; if the download times out, add `--default-timeout=120`)
   - Apple-silicon Mac: `python3 -m pip install torch`
   - AMD / Intel / no dedicated GPU: skip; only the CPU is measured.

## Running

Plug in the charger, set the power mode to best performance and close other apps.

- `python calibrate.py compute`: 3 times, about 15 seconds apart. Use the median.
- `python calibrate.py net`: 2 times. Note Wi-Fi or Ethernet.

Then add the JSON file (and the printed results) to `measurements/` and commit, with the laptop
model, GPU, RAM, connection type and power mode.

## Game frame times

Capture a log with PresentMon or CapFrameX while playing (V-Sync off, so the display refresh rate
does not cap the frame rate), then run `python calibrate.py frames path/to/log.csv`.

## Results so far

See `measurements/MEASUREMENTS.md`: RTX 5070 Ti laptop 1.304 ms per unit, GTX 1660 Ti 3.811 ms per
unit (ratio 0.342); home network 22 ms RTT, 7 ms jitter, 44 Mbps up, about 150 Mbps down.
