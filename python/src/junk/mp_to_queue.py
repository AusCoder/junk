import queue
import multiprocessing
from concurrent import futures


def doesnt_work():
    q = multiprocessing.Queue()

    def _run():
        q.put(1)

    with futures.ProcessPoolExecutor(max_workers=3) as pool:
        fs = [pool.submit(_run) for _ in range(10)]
        done_not_done = futures.wait(fs)

        if any(f.exception() for f in done_not_done.done):
            print([f.exception() for f in done_not_done.done])

    while True:
        try:
            x = q.get(timeout=0.5)
            print(f"Got: {x}")
        except queue.Empty:
            print("Stopped waiting")
            break


if __name__ == "__main__":
    doesnt_work()
