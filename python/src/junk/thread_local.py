import time
import threading
from concurrent import futures


def thread_local_with_vanilla_threads():
    thread_local = threading.local()

    def _run():
        thread_local.count = 1
        print(threading.get_ident(), thread_local.count)

    threads = [threading.Thread(target=_run) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(threading.get_ident(), getattr(thread_local, "count", None))


def thread_local_with_pool():
    thread_local = threading.local()

    def _run(x):
        if hasattr(thread_local, "count"):
            thread_local.count += 1
        else:
            thread_local.count = 1
        print(threading.get_ident(), thread_local.count)
        time.sleep(0.005)

    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        pool.map(_run, range(10))

    print(threading.get_ident(), getattr(thread_local, "count", None))


if __name__ == "__main__":
    # thread_local_with_vanilla_threads()
    thread_local_with_pool()
