import itertools

import pytest

from problems.heap_nodes import heap_insert, heap_pop, BinaryNode


def _gen_values_from_heap(root):
    while root is not None:
        value, root = heap_pop(root)
        yield value


def _gen_samples_for_size(n):
    return ((list(range(n)), p) for p in itertools.permutations(range(n)))


EXAMPLES = itertools.chain.from_iterable(_gen_samples_for_size(n) for n in range(1, 6))


@pytest.mark.parametrize("expected, values", EXAMPLES)
def test_heap_insert_and_pop(expected, values):
    h, *rest = values
    root = BinaryNode(h, None, None)
    for value in rest:
        root = heap_insert(root, value)
    assert expected == list(_gen_values_from_heap(root))
