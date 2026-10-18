# Measured values (calibrate.py) and how the model numbers were derived

Raw data: `zephyrus_rtx5070ti.json`, `predator_acer_gtx1660ti.json` (this folder).
Tool: `..\calibrate.py`. Work unit = one 2048x2048 fp32 matrix multiply (17.18 GFLOP).

## Device speed (GPU, 3 runs each, same script, 2 s warm-up, plugged in)
| Laptop | GPU | ms per unit (median) | units/ms | TFLOPS |
|---|---|---|---|---|
| ASUS Zephyrus G14 | RTX 5070 Ti Laptop | 1.304 (runs 1.296 / 1.317 / 1.304) | 0.7669 | 13.2 |
| Acer Predator | GTX 1660 Ti | 3.811 (runs 3.831 / 3.789 / 3.811) | 0.2624 | 4.5 |

Ratio Predator / Zephyrus = 0.2624 / 0.7669 = 0.342.
Model speeds: current device = 10 (Zephyrus as reference), **aging device = 3.4** (Predator).
CPU is far slower than GPU on both (about 31 ms vs 1.3 ms; about 58 ms vs 3.8 ms), so the GPU sets device speed.
Edge server speed 25 is an ASSUMPTION (2.5x the Zephyrus, about 33 TFLOPS-equivalent); not measured. Needs a spec-sheet citation.
Ignored: early Zephyrus runs with a short warm-up (1.18 ms) and one cold-start outlier (5.36 ms).

## Network (TCP-connect timing to 1.1.1.1 and 8.8.8.8; Cloudflare speed endpoints), 4 runs, 2 laptops
Best-server latency per run: 25.0, 28.2, 19.2, 16.8 ms (Zephyrus, Predator, Zephyrus, Predator).
| Quantity | Runs | Median used |
|---|---|---|
| Latency (RTT) | 25.0 / 28.2 / 19.2 / 16.8 ms | 22 ms |
| Jitter (std of RTT) | 10.7 / 7.2 / 6.8 / 4.9 ms | 7 ms |
| Upload | 49.7 / 39.1 / 44.1 / 44.1 Mbps | 44 Mbps |
| Download | 88.3 / 142.9 / 251.5 / 159.7 Mbps | about 150 Mbps |
| Loss | 1 failed connect in 400 | 0.3% (rough) |
Model: `MEASURED_NET = Network("measured", latency_ms=22, bandwidth_mbps=44, jitter_ms=7, loss_rate=0.003, bandwidth_down_mbps=150)`.
Caveats: home internet path to CDN/anycast servers, NOT a real edge deployment; 1.1.1.1 was consistently slower (32-37 ms) than 8.8.8.8 (17-28 ms); loss estimate is rough.

## Not measured yet
Real game frame times (PresentMon/CapFrameX CSV) to replace SIGMA = 0.3 and the workload means (50/80/120). Run: `python calibrate.py frames log.csv`. Capture with V-Sync off (a 100 Hz display would cap at 100 fps).
