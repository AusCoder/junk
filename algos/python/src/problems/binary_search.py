
def binary_search(sorted_array, target):
    """
    Helpful:
        if I am assigning something like
            if elem < target:
                start_idx = idx
            elif elem > target:
                end_idx = idx
        it is useful to think about the case where
        start_idx and end_idx will stay the same.

    TODO:
        - what to do if no match?
    """
    if len(sorted_array) == 0:
        return None
    if len(sorted_array) == 1:
        return sorted_array[0] if sorted_array[0] == target else None

    start_idx = 0
    end_idx = len(sorted_array) - 1

    while True:
        idx = start_idx + (end_idx - start_idx) // 2
        elem = sorted_array[idx]
        if idx == start_idx or idx == end_idx:
            if sorted_array[start_idx] == target:
                return sorted_array[start_idx]
            if sorted_array[end_idx] == target:
                return sorted_array[end_idx]
            break

        if elem < target:
            start_idx = idx
        elif elem > target:
            end_idx = idx
        else:
            return elem
    return None


if __name__ == "__main__":
    print(binary_search([1, 2, 3, 4, 5], 5))
