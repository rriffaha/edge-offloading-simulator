"""Small, deterministic integration/sanity checks for simulate()."""
import pytest
from model import Device, Job, Network
from sim import policy_edge, policy_local, simulate


def jobs(n=5, work=10):
    return [Job(f"frame {i}", work=work, data_mb=0.2,
                deadline_ms=16.67, arrival_ms=i * 16.67)
            for i in range(n)]


def test_fast_local_device_finishes_every_frame():
    result = simulate(jobs(), Device("fast", speed=10),
                      Network("stable", latency_ms=4, bandwidth_mbps=100),
                      qe_mean=0, policy=policy_local, seed=1)
    assert result.miss_rate == 0
    assert result.drop_rate == 0
    assert result.pct_offloaded == 0
    assert result.data_sent_mb == 0
    assert result.edge_ms == 0
    assert result.avg_completion_ms == pytest.approx(1.0)


def test_lossless_edge_sends_input_and_output_for_every_frame():
    net = Network("stable", latency_ms=4, bandwidth_mbps=100,
                  bandwidth_down_mbps=200, jitter_ms=0, loss_rate=0)
    result = simulate(jobs(), Device("local", speed=10), net,
                      qe_mean=0, policy=policy_edge, seed=1)
    assert result.pct_offloaded == 1
    assert result.drop_rate == 0
    assert result.miss_rate == 0
    assert result.data_sent_mb == pytest.approx(5 * (0.2 + 0.4))
    assert result.edge_ms == pytest.approx(5 * 10 / 25)


def test_all_packets_lost_count_as_drops_and_misses():
    net = Network("lossy", latency_ms=4, bandwidth_mbps=100, loss_rate=1)
    result = simulate(jobs(), Device("local", speed=10), net,
                      qe_mean=0, policy=policy_edge, seed=7)
    assert result.pct_offloaded == 1
    assert result.drop_rate == 1
    assert result.miss_rate == 1
    assert result.data_sent_mb == pytest.approx(5 * 0.2)
    assert result.edge_ms == 0


def test_same_seed_gives_identical_metrics():
    workload = jobs(20, work=30)
    net = Network("varying", latency_ms=12, bandwidth_mbps=50,
                  jitter_ms=3, loss_rate=0.1)
    first = simulate(workload, Device("local", speed=3), net,
                     qe_mean=3, policy=policy_edge, seed=55)
    second = simulate(workload, Device("local", speed=3), net,
                      qe_mean=3, policy=policy_edge, seed=55)
    assert first == second


def test_sanity_bounds():
    result = simulate(jobs(20, work=50), Device("local", speed=5),
                      Network("network", latency_ms=8, bandwidth_mbps=50),
                      qe_mean=1, policy=policy_local, seed=3)
    assert 0 <= result.drop_rate <= result.miss_rate <= 1
    assert 0 <= result.pct_offloaded <= 1
    assert result.edge_ms >= 0
    assert result.data_sent_mb >= 0
