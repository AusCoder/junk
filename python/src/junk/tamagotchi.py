import time

from junk.keyboard_non_blocking import NonBlockingKeyboard

TICK_DURATION = 0.05

INITIAL_FOOD_LEVEL = 100
FOOD_PER_TICK = -1
FOOD_PER_FEED = 10
MAX_FOOD_LEVEL = 100

INITIAL_ENERGY_LEVEL = 50
ENERGY_PER_TICK_AWAKE = -1
ENERGY_PER_TICK_ASLEEP = 5
MAX_ENERGY_LEVEL = 100

INITIAL_IS_AWAKE = False

INITIAL_POOP_LEVEL = 0
TICKS_PER_POOP = 25
MAX_POOP_LEVEL = 10


class UnknownCommand(Exception):
    pass


def _add_and_clip(x, dx, x_min, x_max):
    return max(x_min, min(x_max, x + dx))


class Tamagotchi:
    def __init__(self) -> None:
        self._age = 0
        self._food_level = INITIAL_FOOD_LEVEL
        self._energy_level = INITIAL_ENERGY_LEVEL
        self._poop_level = INITIAL_POOP_LEVEL
        self._is_awake = INITIAL_IS_AWAKE
        self._commands = {
            "f": self._feed,
            "c": self._clean,
            "s": self._sleep,
        }

    def __repr__(self) -> str:
        return f"Tamagotchi(is_awake={self._is_awake}, food_level={self._food_level}, energy_level={self._energy_level}, poop_level={self._poop_level}, age={self._age})"

    def process_command(self, command: str) -> None:
        try:
            self._commands[command]()
        except KeyError:
            raise UnknownCommand(command)

    def _feed(self) -> None:
        if self._is_awake:
            self._food_level = _add_and_clip(
                self._food_level, FOOD_PER_FEED, 0, MAX_FOOD_LEVEL
            )

    def _clean(self) -> None:
        self._poop_level = 0

    def _sleep(self) -> None:
        self._is_awake = False

    def is_alive(self) -> bool:
        return self._food_level > 0 and self._poop_level < MAX_POOP_LEVEL

    def update(self) -> None:
        self._age += 1
        # Food
        self._food_level = _add_and_clip(
            self._food_level, FOOD_PER_TICK, 0, MAX_FOOD_LEVEL
        )
        # Energy
        if self._energy_level >= MAX_ENERGY_LEVEL:
            self._is_awake = True
        if self._energy_level <= 0:
            self._is_awake = False
        energy_delta = (
            ENERGY_PER_TICK_AWAKE if self._is_awake else ENERGY_PER_TICK_ASLEEP
        )
        self._energy_level = _add_and_clip(
            self._energy_level, energy_delta, 0, MAX_ENERGY_LEVEL
        )
        # Poop
        if self._age % TICKS_PER_POOP == 0:
            self._poop_level += 1


def main():
    tamagotchi = Tamagotchi()
    with NonBlockingKeyboard() as kb:
        while True:
            inpt = kb.getstr()

            should_quit = False
            for c in inpt:
                try:
                    tamagotchi.process_command(c)
                except UnknownCommand:
                    if c == "q":
                        should_quit = True
                        break
                    else:
                        raise

            if should_quit:
                break

            tamagotchi.update()
            print(tamagotchi)
            if not tamagotchi.is_alive():
                print("tamagotchi died")
                break
            time.sleep(TICK_DURATION)


if __name__ == "__main__":
    main()
