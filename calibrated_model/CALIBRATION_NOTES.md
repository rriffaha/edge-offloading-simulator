# Calibration notes (measured with calibrate.py)

Device speed (1 work unit = one 2048x2048 fp32 matmul, GPU):
- RTX 5070 Ti Laptop (Zephyrus): 1.296 / 1.317 / 1.304 ms per unit (3 runs, ~13.2 TFLOPS) -> reference, speed 10.
- GTX 1660 Ti (Acer Predator): 3.83 / 3.79 / 3.81 ms per unit (3 runs, ~4.5 TFLOPS) -> 34.2% of reference -> speed 3.4.
- Both measured with the same calibrate.py (2 s GPU warm-up, sustained speed). Earlier Zephyrus runs with a shorter warm-up (1.18 ms) were discarded.
- Edge server speed (25) is still an ASSUMPTION (not measurable here); cite a GPU spec sheet.

Network (TCP-connect timing to 1.1.1.1 / 8.8.8.8, Cloudflare speed endpoints; 2 laptops, 4 runs):
- best latency 25, 28, 19, 17 ms (median 22); jitter 10.7, 7.2, 6.8, 4.9 ms (median 7)
- upload 50, 39, 44, 44 Mbps (median 44); download 88, 143, 251, 160 Mbps (median ~150)
- loss: 1 failed connection in 400 samples (~0.3%, rough)
- Home-internet path, NOT a real edge deployment. good/moderate/poor placeholders kept as edge scenarios.

Results: results_calibrated.csv (192 rows = 2 devices x 3 workloads x 4 networks x 2 congestion x 4 strategies, 20 seeds).
Run from this folder: python run_experiments.py
