class BinaryNode:
    def __init__(self, value, left, right, parent=None):
        self.value = value
        self.left = left
        self.right = right
        self.parent = parent

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value}, left={self.left}, right={self.right})"

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value

    def depth(self):
        if self.left is None and self.right is None:
            return 1
        elif self.left is None:
            return 1 + self.right.depth()
        elif self.right is None:
            return 1 + self.left.depth()
        else:
            return 1 + max(self.left.depth(), self.right.depth())


class Node:
    def __init__(self, value=None, children=None):
        children = children if children else []
        self.value = value
        self.children = children
        # Useful properties used by algorithms
        self.visited = False
        self.distance = None
        self.in_degree = None

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value}, visited={self.visited}, distance={self.distance}, in_degree={self.in_degree})"

    @property
    def out_degree(self):
        return len(self.children)
