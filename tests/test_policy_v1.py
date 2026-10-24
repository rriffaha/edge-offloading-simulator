"""Tests for the deadline-aware v1 rule. True means offload."""
import pytest
from sim import MARGIN_MS, policy_v1


@pytest.mark.parametrize("local_ms,edge_ms,deadline_ms,expected", [
    (10, 5, 16.67, False),  # local on time: keep locally even if edge faster
    (17, 8, 16.67, True),   # local misses; edge safely meets deadline
    (20, 18, 16.67, True),  # neither meets deadline; edge is faster
    (18, 20, 16.67, False), # neither meets deadline; local is faster
    (20, 20, 16.67, False), # equal estimates: local wins tie
])
def test_policy_v1_choices(local_ms, edge_ms, deadline_ms, expected):
    assert policy_v1(local_ms, edge_ms, deadline_ms, []) is expected


def test_policy_v1_local_at_safety_margin_boundary():
    d = 16.67
    assert policy_v1(d - MARGIN_MS, 1, d, []) is False
    assert policy_v1(d - MARGIN_MS + 0.01, 1, d, []) is True
