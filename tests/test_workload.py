"""Tests for frame arrivals, job construction and seeded workloads."""
import pytest
from model import FRAME_DEADLINE_MS
from workload import FRAME_DATA_MB, PRESETS, generate_jobs


def test_generated_jobs_have_correct_count_and_arrivals():
    jobs = generate_jobs(10, PRESETS["moderate"], 0.3, seed=1)
    assert len(jobs) == 10
    for i, job in enumerate(jobs):
        assert job.name == f"frame {i}"
        assert job.arrival_ms == pytest.approx(i * FRAME_DEADLINE_MS)
        assert job.deadline_ms == pytest.approx(FRAME_DEADLINE_MS)
        assert job.data_mb == pytest.approx(FRAME_DATA_MB)
        assert job.work > 0


def test_same_seed_reproduces_workloads():
    assert generate_jobs(50, 80, 0.3, seed=42) == generate_jobs(50, 80, 0.3, seed=42)


def test_different_seeds_produce_different_workloads():
    a = generate_jobs(50, 80, 0.3, seed=1)
    b = generate_jobs(50, 80, 0.3, seed=2)
    assert [j.work for j in a] != [j.work for j in b]


def test_zero_sigma_produces_constant_work():
    jobs = generate_jobs(8, 75, 0.0, seed=10)
    assert all(j.work == pytest.approx(75) for j in jobs)


def test_zero_frames_returns_empty_list():
    assert generate_jobs(0, 80, 0.3, seed=1) == []
