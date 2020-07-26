"""Saw this problem on youtube.
You can walk up a set of stairs either taking 1 or 2 steps at a time.
The problem is to count the number of ways to walk up the stairs.

It took me too long to realise this is generating the fibonacci sequence.
"""
from collections import deque


def count_ways(n):
    """Counts the number of ways to walk up stairs with n steps
    """
    # return _count_ways_with_cache(n, {})
    return _count_ways_generated(n)


def _count_ways_with_cache(n, cache):
    """This hits a recursion limit after 1000
    Can we use a stack?
    """
    if n in cache:
        return cache[n]
    if n <= 0:
        result = 0
    elif n == 1:
        result = 1
    elif n == 2:
        result = 2
    else:
        result = _count_ways_with_cache(n - 1, cache) + _count_ways_with_cache(
            n - 2, cache
        )
    cache[n] = result
    return result


def _count_ways_generated(n):
    if n <= 0:
        return 0

    x = 1
    y = 2
    for _ in range(n - 1):
        t = y
        y = x + y
        x = t
    return x


if __name__ == "__main__":
    print(count_ways(1000))
