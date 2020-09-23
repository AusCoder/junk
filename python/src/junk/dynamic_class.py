def make_class(x: int) -> type:
    class ValueHolder:
        def __init__(self, value: int = x) -> None:
            self.value = value

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}(value={self.value})"

    return ValueHolder


if __name__ == "__main__":
    cls1 = make_class(1)
    x = cls1()
    print(x)
    cls2 = make_class(2)
    x = cls2()
    print(x)

    try:
        ValueHolder
        raise RuntimeError
    except NameError:
        pass
