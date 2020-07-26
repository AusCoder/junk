import pytest

from problems.nodes import Node
from problems.topological_sort import topological_sort


def test_topological_sort():
    n_1 = Node(value="n_1")
    n_0 = Node(value="n_0", children=[n_1])
    n1 = Node(value="n1", children=[n_0, n_1])
    n2 = Node(value="n2", children=[n_0])
    n3 = Node(value="n3", children=[n1])
    n4 = Node(value="n4", children=[n1, n2])
    nodes = [
        n_1,
        n_0,
        n1,
        n2,
        n3,
        n4,
    ]
    top_sorted = topological_sort(nodes)
    assert top_sorted[0].value in ["n4", "n3"]
    assert top_sorted[-1].value == "n_1"


def test_topological_sort_cycle():
    n0 = Node(value="n0")
    n1 = Node(value="n1", children=[n0])
    n2 = Node(value="n2", children=[n1])
    n3 = Node(value="n3", children=[n2])
    n0.children.append(n3)
    n4 = Node(value="n4", children=[n0])
    nodes = [n0, n1, n2, n3, n4]
    with pytest.raises(ValueError):
        topological_sort(nodes)


def test_topological_sort_empty():
    assert topological_sort([]) == []


def test_topological_sort_single():
    n = Node(value=1)
    assert topological_sort([n]) == [n]
