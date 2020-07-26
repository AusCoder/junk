"""Naievly builds a binary search tree from a list of elements.
The tree will almost certainly be unbalanced.

Follow up questions:
- [X] Assume the elements are sorted, write an algorithm to make a balanced binary search tree.
        Ie one that has minimal height.
- [ ] Implement the AVL algorithm here too.
"""
from problems.nodes import BinaryNode


def build_binary_search_tree_unbalanced(elements):
    try:
        head, *rest = elements
    except ValueError as err:
        raise RuntimeError("No elements to build binary search tree") from err

    root = BinaryNode(head, None, None)
    for value in rest:
        node = BinaryNode(value, None, None)
        _insert_node(root, node)
    return root


def _insert_node(root, node):
    if node < root and root.left is None:
        root.left = node
    elif node < root:
        _insert_node(root.left, node)
    elif node > root and root.right is None:
        root.right = node
    elif node > root:
        _insert_node(root.right, node)
    else:
        raise RuntimeError(
            "Case of duplicated value in binary search tree not implemented"
        )


def build_binary_search_tree_from_sorted(elements):
    if not elements:
        raise RuntimeError
    if len(elements) == 1:
        return BinaryNode(elements[0], None, None)
    if len(elements) == 2:
        return BinaryNode(
            elements[0], None, build_binary_search_tree_from_sorted(elements[1:])
        )
    mid_idx = len(elements) // 2
    left = build_binary_search_tree_from_sorted(elements[:mid_idx])
    right = build_binary_search_tree_from_sorted(elements[mid_idx + 1 :])
    return BinaryNode(elements[mid_idx], left, right)


if __name__ == "__main__":
    # root = build_binary_search_tree_unbalanced([2, 1, 3])
    # print(root)

    root = build_binary_search_tree_from_sorted(list(range(8)))
    print(root)
    print(root.depth())
