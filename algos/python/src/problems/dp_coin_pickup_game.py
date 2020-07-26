"""Dynamic programming coin choose game
"""

import operator
import functools


def main():
    # coins = tuple([5, 3, 7, 10])
    coins = tuple([8, 15, 3, 7])
    print(optimal_play(coins))


@functools.lru_cache
def optimal_play(coins):
    try:
        h, *rest, t = coins
    except ValueError:
        (h,) = coins
        return h, 0
    # I can choose h or t
    # I want to let the opponent play, then take the option that minimizes the
    # opponents value.
    # It isn't obvious to me why taking the option that minimizes the opponents score
    # works, while the taking the option that maximizes my total doesn't work.
    sub_plays = [
        (*optimal_play(tuple([*rest, t])), h),
        (*optimal_play(tuple([h, *rest])), t),
    ]
    # Option that minimizes opponents score - works
    op, my, c = min(sub_plays, key=operator.itemgetter(0))
    # Option that maximizes my score - doesn't work
    # op, my, c = max(sub_plays, key=operator.itemgetter(1))
    return my + c, op


if __name__ == "__main__":
    main()
