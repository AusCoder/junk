"""
100 prisoners go into a room, each is numbered, in the room
is a series of draws with their numbers randomly shuffled in them.
Find a strategy where there is good odds of them all getting
their number.
"""
import math
import itertools


def tiled_is_valid_perm(perm):
    n = len(perm)
    assert n % 2 == 0
    half = n // 2

    def is_idx_valid(i):
        for j in range(half):
            if i == perm[(i + j) % n]:
                return True
        return False

    return all(is_idx_valid(i) for i in perm)


def two_groups_is_valid_perm(perm):
    n = len(perm)
    assert n % 2 == 0
    half = n // 2

    def is_idx_valid(i):
        if i < half:
            return i in perm[:half]
        else:
            return i in perm[half:]

    return all(is_idx_valid(i) for i in perm)


# fat_tiled
def double_tiled_is_valid_perm(perm):
    n = len(perm)
    assert n % 2 == 0
    half = n // 2

    def is_idx_valid(i):
        start = i - i % 2
        for j in range(half):
            if i == perm[(start + j) % n]:
                return True
        return False

    return all(is_idx_valid(i) for i in perm)


# # Something like lots of tranpositions
# def is_valid_perm(perm):
#     n = len(perm)
#     assert n % 2 == 0
#     half = n // 2

#     def is_idx_valid(i):
#         start = i - i % 2
#         found = False
#         for j in range(half):
#             if i == perm[(start + j) % n]:
#                 found = True
#         return found

#     return all([is_idx_valid(i) for i in perm])


def transform_perm(perm):
    return [x + 1 for x in perm]


if __name__ == "__main__":
    # is_valid_perm(range(6))
    ns = [4, 6, 8, 10]
    for n in ns:
        nums = list(range(n))
        perms = itertools.permutations(nums)
        num_valid_perms = sum(1 for perm in perms if is_valid_perm(perm))
        prob = num_valid_perms / math.factorial(n)
        print(f"n: {n} num valid perms: {num_valid_perms} prob {prob:.6f}")

    # nums = list(range(6))
    # perms = itertools.permutations(nums)
    # valid_perms = [perm for perm in perms if is_valid_perm(perm)]
    # valid_perms = [transform_perm(perm) for perm in valid_perms]
    # for perm in valid_perms:
    #     print(perm)
