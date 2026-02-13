"""Tests for GPU dispatch layer."""
import pytest


def test_cuda_available():
    from angle_only import gpu

    avail = gpu.cuda_available()
    count = gpu.device_count()
    assert isinstance(avail, bool)
    assert count >= 0
    if not avail:
        assert count == 0


def test_should_use_gpu():
    from angle_only import gpu

    # Below threshold should be False (no GPU on test machine)
    assert not gpu.should_use_gpu(1)
