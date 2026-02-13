"""Tests for data association."""
import pytest
import numpy as np


def test_gnn_assign():
    from angle_only import association

    cost = np.array([[1, 10, 10],
                     [10, 2, 10],
                     [10, 10, 3]], dtype=float)
    result = association.gnn_assign(cost)
    assert len(result.assignments) == 3
    assert abs(result.total_cost - 6.0) < 1e-10


def test_gate():
    from angle_only import association

    pred = np.array([0.0, 0.0])
    measurements = [np.array([0.1, 0.0]), np.array([10.0, 10.0])]
    S = np.eye(2)
    gated = association.gate(pred, measurements, S, 1.0)
    assert 0 in gated
    assert 1 not in gated


def test_jpda():
    from angle_only import association

    lik = np.array([[0.8, 0.1], [0.1, 0.7]])
    result = association.jpda_probabilities(lik)
    for i in range(result.beta.shape[0]):
        assert abs(result.beta[i, :].sum() - 1.0) < 1e-10
