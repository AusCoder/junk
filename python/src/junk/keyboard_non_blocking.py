import os
import sys
import termios
import select
import time
from typing import Generator


class NonBlockingKeyboard:
    def __init__(self) -> None:
        self._file = sys.stdin
        self._old_settings = None

    def __enter__(self):
        self._old_settings = termios.tcgetattr(self._file)
        cur_settings = termios.tcgetattr(self._file)
        cur_settings[3] = cur_settings[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(self._file, termios.TCSAFLUSH, cur_settings)
        return self

    def __exit__(self, *args, **kwargs):
        termios.tcsetattr(self._file, termios.TCSAFLUSH, self._old_settings)

    def getstr(self) -> str:
        return "".join(self._gen_chars())

    def _gen_chars(self) -> Generator[str, None, None]:
        while True:
            readable, _, _ = select.select([self._file], [], [], 0)
            if readable:
                yield self._file.read(1)
            else:
                break


if __name__ == "__main__":
    with NonBlockingKeyboard() as kb:
        while True:
            s = kb.getstr()
            if s:
                print(f"Received: '{s}'")
            time.sleep(0.05)
