"""Implements topological sort on a DAG
"""
from collections import deque

from problems.nodes import Node


def topological_sort(nodes):
    for node in nodes:
        node.in_degree = 0
    for node in nodes:
        for child_node in node.children:
            child_node.in_degree += 1

    top_sorted = []
    q = deque()
    for node in nodes:
        if node.in_degree == 0:
            q.append(node)
    while q:
        cur_node = q.popleft()
        top_sorted.append(cur_node)
        for child_node in cur_node.children:
            child_node.in_degree -= 1
            if child_node.in_degree == 0:
                q.append(child_node)

    if len(top_sorted) != len(nodes):
        raise ValueError("Graph provided contains a cycle")

    return top_sorted
