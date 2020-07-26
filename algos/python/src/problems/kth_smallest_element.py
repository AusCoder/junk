"""Find the kth smallest element of a random list of elems
"""
import heapq
import random


def kth_smallest_heap(k, elems):
    heap = []
    for elem in elems:
        if len(heap) < k:
            heapq.heappush(heap, elem)
        else:
            heapq.heappushpop(heap, elem)
    return heap[0]


def kth_smallest_lists(k, elems):
    largest_elems = []
    for elem in elems:  # O(len(elems))
        if len(largest_elems) < k:
            largest_elems.append(elem)
        else:
            current_min = min(largest_elems)  # O(k)
            if elem > current_min:
                largest_elems.remove(current_min)
                largest_elems.append(elem)

    return min(largest_elems)


def main():
    k = 4
    elems = [random.randint(0, 100) for _ in range(50)]

    print(kth_smallest_heap(k, elems))
    print(kth_smallest_lists(k, elems))


if __name__ == "__main__":
    main()
