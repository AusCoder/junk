class CircularArray:
    def __init__(self, size):
        self.elems = [None] * size
        self.rotation = 0

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rotation={self.rotation}, elems={self.elems})"
        )

    # Rotation
    def rotate(self, rotation):
        self.rotation += rotation
        self.rotation %= len(self)

    # Array api
    # indexing
    def __getitem__(self, idx):
        return self.elems[self._get_transformed_idx(idx)]

    def __setitem__(self, idx, value):
        self.elems[self._get_transformed_idx(idx)] = value

    def _get_transformed_idx(self, idx):
        return (self.rotation + idx) % len(self)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    # size
    def __len__(self):
        return len(self.elems)


if __name__ == "__main__":
    arr = CircularArray(3)
    arr.rotate(1)
    arr[0] = 3
    arr.rotate(-1)
    arr[0] = 2
    arr.rotate(-1)
    arr[0] = 1
    print(arr)
    print(list(arr))
