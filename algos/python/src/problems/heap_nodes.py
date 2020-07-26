"""Heaps using nodes"""
from collections import deque

from problems.nodes import BinaryNode


def heap_insert(root, value):
    new_node = BinaryNode(value, None, None)
    last_elem = _get_insert_position(root)

    new_node.parent = last_elem
    if last_elem.left is None:
        last_elem.left = new_node
    elif last_elem.right is None:
        last_elem.right = new_node
    else:
        raise RuntimeError

    _heapify_from_base(new_node)
    return _heap_get_root(root)


def _heapify_from_base(base):
    parent = base.parent
    if parent is not None and base.value < parent.value:
        t = base.value
        base.value = parent.value
        parent.value = t
        _heapify_from_base(parent)


def _heap_get_root(node):
    while node.parent is not None:
        node = node.parent
    return node


def _get_insert_position(root):
    """Use a bfs to get the next node for insert
    This is unfortunate because it means traversing the
    whole heap to find the new position to insert.
    """
    q = deque()
    q.append(root)
    while q:
        n = q.popleft()
        if n.left is None or n.right is None:
            return n
        q.append(n.left)
        q.append(n.right)
    raise RuntimeError


def heap_pop(root):
    ret = root.value
    right_most = _get_rightmost_element(root)
    if right_most.parent is None:
        return ret, None
    root.value = right_most.value
    # Remove the right_most node
    if right_most.parent.left is right_most:
        right_most.parent.left = None
    else:
        right_most.parent.right = None
    right_most.parent = None

    _heapify_from_root(root)
    return ret, _heap_get_root(root)


def _heapify_from_root(root):
    if root.left is not None and root.right is not None:
        min_child = min(root.left, root.right)
    elif root.left is not None:
        min_child = root.left
    elif root.right is not None:
        min_child = root.right
    else:
        return
    if root > min_child:
        t = min_child.value
        min_child.value = root.value
        root.value = t
        _heapify_from_root(min_child)


def _get_rightmost_element(root):
    while True:
        if root.left is None:
            return root
        if root.right is None:
            return _get_rightmost_element(root.left)
        root = root.right


if __name__ == "__main__":
    root = BinaryNode(5, None, None)
    print(root)
    root = heap_insert(root, 3)
    print(root)
    root = heap_insert(root, 6)
    print(root)
    root = heap_insert(root, 4)
    print(root)
    root = heap_insert(root, 7)
    print(root)
    root = heap_insert(root, 2)
    print(root)
    root = heap_insert(root, 1)
    print(root)

    while root is not None:
        min_value, root = heap_pop(root)
        print(min_value)
        print(root)
        print()
