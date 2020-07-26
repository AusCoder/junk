from collections import deque
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Node:
    index: int
    neighbours: List["Node"]
    marked: bool = False

    def __repr__(self):
        return f"{self.__class__.__name__}(index={self.index})"


class GraphTraverseStructure:
    def __init__(self):
        self._deque = deque()

    def __len__(self):
        return len(self._deque)

    def push(self, x):
        self._deque.append(x)

    def pop(self):
        raise NotImplementedError


class StackGraphTraverseStructure(GraphTraverseStructure):
    def pop(self):
        return self._deque.pop()


class QueueGraphTraverseStructure(GraphTraverseStructure):
    def pop(self):
        return self._deque.popleft()


def _traverse(traversal_structure_cls: Any):
    def traverse(start: Node, fn: Any) -> None:
        structure = traversal_structure_cls()
        start.marked = True
        structure.push(start)

        while structure:
            current = structure.pop()
            fn(current)
            ns = sorted(current.neighbours, key=lambda n: n.index)
            for n in ns:
                if not n.marked:
                    n.marked = True
                    structure.push(n)

    return traverse


def traverse_dfs(start: Node, fn: Any) -> None:
    return _traverse(StackGraphTraverseStructure)(start, fn)


def traverse_bfs(start: Node, fn: Any) -> None:
    return _traverse(QueueGraphTraverseStructure)(start, fn)


if __name__ == "__main__":
    def example1():
        node0 = Node(0, [])
        node1 = Node(1, [])
        node2 = Node(2, [])
        node3 = Node(3, [])
        node4 = Node(4, [])
        node5 = Node(5, [])

        node0.neighbours.append(node1)
        node0.neighbours.append(node4)
        node0.neighbours.append(node5)

        node1.neighbours.append(node3)
        node1.neighbours.append(node4)

        node2.neighbours.append(node1)

        node3.neighbours.append(node2)
        node3.neighbours.append(node4)

        traverse_bfs(node0, print)


    def example2():
        node0 = Node(0, [])
        node1 = Node(1, [])
        node2 = Node(2, [])
        node3 = Node(3, [])
        node4 = Node(4, [])
        node5 = Node(5, [])

        node0.neighbours.append(node1)
        node1.neighbours.append(node2)

        node0.neighbours.append(node3)
        node3.neighbours.append(node4)

        node2.neighbours.append(node5)
        node4.neighbours.append(node5)

        traverse_dfs(node0, print)

    example2()
