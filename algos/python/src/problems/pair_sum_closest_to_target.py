"""Given 2 arrays of ints and a target, find the pair whose sum is closest to the target.
"""
import itertools


def closest_to_target_brute_force(array1, array2, target):
    best_pair = None
    best_distance = None
    for elem1, elem2 in itertools.product(array1, array2):
        if best_pair is None:
            best_pair = (elem1, elem2)
            best_distance = abs(elem1 + elem2 - target)
            continue

        current_distance = abs(elem1 + elem2 - target)
        if current_distance < best_distance:
            best_pair = (elem1, elem2)
            best_distance = current_distance
    return best_pair


def closest_to_target_linear(array1, array2, target):
    """Idea:
    - Have pointers running along both sorted arrays in opposite directions.

    This idea of having pointers running in opposite directions is pretty
    powerful.
    """
    array1 = sorted(array1)
    array2 = sorted(array2)
    idx1 = 0
    idx2 = len(array2) - 1
    best_pair = (array1[idx1], array2[idx2])
    best_distance = abs(array1[idx1] + array2[idx2] - target)

    while idx1 < len(array1) and idx2 >= 0:
        current_distance = array1[idx1] + array2[idx2] - target
        if abs(current_distance) < best_distance:
            best_distance = abs(current_distance)
            best_pair = (array1[idx1], array2[idx2])

        if current_distance < 0:
            idx1 += 1
        elif current_distance > 0:
            idx2 -= 1
        else:
            break
    return best_pair


def main():
    array1 = [-1, 3, 8, 2, 9, 5]
    array2 = [4, 1, 2, 10, 5, 20]
    target = 24

    print(closest_to_target_brute_force(array1, array2, target))
    print(closest_to_target_linear(array1, array2, target))


if __name__ == "__main__":
    main()
