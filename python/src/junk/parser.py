import functools


# Push based parser parser


class PushParser:
    def __init__(self, callback):
        self.callback = callback
        self._values = []
        self._parsers = [
            parse_list,
            parse_int,
            functools.partial(parse_str, "abc"),
            functools.partial(parse_str, "def"),
        ]

    def accept(self, chunk):
        self._values.append(chunk)
        for parser in self._parsers:
            try:
                parsed_value, values = parser(self._values)
                if parsed_value is not _NEEDS_MORE:
                    self._values = values
                    self.callback(parsed_value)
                return
            except AssertionError:
                pass
        raise AssertionError(f"Unable to parse: {self._values}")


_NEEDS_MORE = object()


def parse_str(s, values):
    head, *rest = values
    assert head == s
    return head, rest


def parse_int(values):
    try:
        head, *rest = values
        return int(head), rest
    except ValueError as err:
        raise AssertionError(values) from err


def parse_list(values):
    head, *rest = values
    if head.startswith("l:"):
        n = int(head.replace("l:", ""))
        if len(rest) < n:
            return _NEEDS_MORE, values
        else:
            parsed_values = []
            for _ in range(n):
                v, rest = parse_int(rest)
                parsed_values.append(v)
            return parsed_values, rest
    else:
        raise AssertionError(values[0])


def callback(value):
    print(f"in callback: {value}")


# Using coroutines
# has similar api to the PushParser


def co(fn):
    def f(*args, **kwargs):
        g = fn(*args, **kwargs)
        g.send(None)
        return g

    return f


@co
def mult(x, co):
    while True:
        y = yield
        co.send(x * y)


@co
def consumer(callback):
    while True:
        y = yield
        callback(y)


@co
def stream_parser(parsers, consumer):
    values = []
    while True:
        x = yield
        values.append(x)
        did_parse_value = False
        for parser in parsers:
            try:
                parsed_value, values = parser(values)
                if parsed_value is not _NEEDS_MORE:
                    consumer.send(parsed_value)
                did_parse_value = True
                break
            except AssertionError:
                pass
        if not did_parse_value:
            raise AssertionError(values)


if __name__ == "__main__":
    stream = [
        "abc",
        "123",
        "l:3",
        "123",
        "456",
        "789",
        "321",
        "def",
    ]
    # push_parser = PushParser(callback)
    # for chunk in stream:
    #     push_parser.accept(chunk)

    cons = consumer(callback)
    parsers = [
        functools.partial(parse_str, "abc"),
        functools.partial(parse_str, "def"),
        parse_int,
        parse_list,
    ]
    par = stream_parser(parsers, cons)
    for chunk in stream:
        par.send(chunk)
