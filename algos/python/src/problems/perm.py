"""Permutations of a sequence"""


def permutations(elements):
    if len(elements) <= 1:
        return [tuple(elements)]
    perms = []
    for i, elem in enumerate(elements):
        rest = elements[:i] + elements[i + 1 :]
        perms.extend((elem,) + sub for sub in permutations(rest))
    return perms


if __name__ == "__main__":
    print(permutations([1, 2, 3, 4]))
    print(len(permutations([1, 2, 3, 4])))
