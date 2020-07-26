import math

import pytest

from problems.binary_search_tree import build_binary_search_tree_from_sorted


@pytest.mark.parametrize("num_elements", range(1, 102))
def test_build_binary_search_tree_from_sorted_depth(num_elements):
    root = build_binary_search_tree_from_sorted(list(range(num_elements)))
    assert root.depth() == math.floor(math.log(num_elements, 2)) + 1


def _is_valid_from_root(root):
    if root.left is None and root.right is None:
        return True
    elif root.left is None:
        return root.value < root.right.value and _is_valid_from_root(root.right)
    elif root.right is None:
        return root.left.value < root.value and _is_valid_from_root(root.left)
    else:
        return (
            root.left.value < root.value < root.right.value
            and _is_valid_from_root(root.left)
            and _is_valid_from_root(root.right)
        )


@pytest.mark.parametrize("num_elements", range(1, 102))
def test_build_binary_search_tree_from_sorted_validity(num_elements):
    root = build_binary_search_tree_from_sorted(list(range(num_elements)))
    assert _is_valid_from_root(root)
