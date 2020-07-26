import itertools

import pytest

from problems.binary_search import binary_search


def _samples_of_size(n):
    return ((list(range(n)), i) for i in range(n))


BINARY_SEARCH_EXAMPLES = itertools.chain.from_iterable(
    _samples_of_size(n) for n in range(10)
)


@pytest.mark.parametrize("sorted_array, target", BINARY_SEARCH_EXAMPLES)
def test_binary_search(sorted_array, target):
    assert binary_search(sorted_array, target) == target
