import pytest

from problems.recursive_stairs import count_ways


@pytest.mark.parametrize(
    "num_stairs, expected", [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5),]
)
def test_count_ways(num_stairs, expected):
    assert count_ways(num_stairs) == expected
