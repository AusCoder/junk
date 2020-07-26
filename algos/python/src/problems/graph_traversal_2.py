"""Another implementation of bfs and dfs.
"""
from collections import deque

from problems.nodes import Node


def bfs_from_node(node):
    q = deque()
    q.append(node)
    node.distance = 0
    while q:
        cur_node = q.popleft()
        if not cur_node.visited:
            print(cur_node)
            cur_node.visited = True
            for child_node in cur_node.children:
                if child_node.distance is None:
                    child_node.distance = cur_node.distance + 1
                q.append(child_node)


def dfs_from_node(node):
    s = list()
    s.append(node)
    while s:
        cur_node = s.pop()
        if not cur_node.visited:
            print(cur_node)
            cur_node.visited = True
            for child_node in cur_node.children:
                s.append(child_node)


if __name__ == "__main__":
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

    dfs_from_node(n4)

    # for n in nodes:
    #     print(n)
