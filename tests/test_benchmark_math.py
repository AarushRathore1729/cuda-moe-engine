import pytest
from moe_dispatch.benchmark import percentile


def test_percentile_uses_linear_interpolation():
    assert percentile([10, 30, 20, 40], 0.95) == pytest.approx(38.5)
    assert percentile([7], 0.95) == 7
    assert percentile([], 0.95) == 0
