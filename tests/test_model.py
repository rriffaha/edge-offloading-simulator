"""Unit tests for the local and edge completion-time equations."""
import pytest
from model import Device, Job, Network, t_local, t_edge


def test_local_time_without_queue():
    job = Job("sample", work=100, data_mb=0.2, deadline_ms=16.67)
    assert t_local(job, Device("gpu", speed=10)) == pytest.approx(10.0)


def test_local_time_adds_queue_delay():
    job = Job("sample", work=100, data_mb=0.2, deadline_ms=16.67)
    assert t_local(job, Device("gpu", speed=10), queue_ms=3) == pytest.approx(13.0)


def test_edge_time_uses_separate_upload_and_download_bandwidth():
    job = Job("sample", work=100, data_mb=0.2, deadline_ms=16.67, data_out_mb=0.4)
    network = Network("test", latency_ms=4, bandwidth_mbps=100, bandwidth_down_mbps=200)
    # 4 ms RTT + 2 ms upload + 3 ms queue + 4 ms compute + 2 ms download.
    assert t_edge(job, network, Device("edge", speed=25), qe_ms=3) == pytest.approx(15.0)


def test_edge_downlink_defaults_to_uplink():
    net = Network("symmetric", latency_ms=4, bandwidth_mbps=50)
    assert net.bandwidth_down_mbps == 50
