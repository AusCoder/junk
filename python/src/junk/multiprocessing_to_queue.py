import time
import queue
import threading
import multiprocessing
from concurrent import futures


def doesnt_work():
    """This is not really the right way to use ProcessPoolExecutor
    or ThreadPoolExecutor. They are meant to handle the queuing
    stuff for you!
    """
    q = multiprocessing.Queue()

    def _run(qu):
        qu.put(1)

    with futures.ProcessPoolExecutor(max_workers=3) as pool:
        fs = [pool.submit(_run, q) for _ in range(10)]

        while True:
            try:
                x = q.get(timeout=0.5)
                print(f"Got: {x}")
            except queue.Empty:
                print("Stopped waiting")
                break

        done_not_done = futures.wait(fs)
        if any(f.exception() for f in done_not_done.done):
            print([f.exception() for f in done_not_done.done])


def to_queue_with_vanilla_processes():
    q = multiprocessing.Queue()

    def _run():
        q.put(1)

    procs = [multiprocessing.Process(target=_run) for _ in range(3)]
    for p in procs:
        p.start()
    while True:
        try:
            x = q.get(timeout=0.5)
            print(f"Got: {x}")
        except queue.Empty:
            print("Stopped waiting")
            break
    for p in procs:
        p.join()


def _gen_chunks(xs, chunksize):
    assert chunksize > 0
    chunk = []
    for x in xs:
        chunk.append(x)
        if len(chunk) == chunksize:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def parallel_process_to_queue(xs, chunksize, q):
    def _done_cb(f):
        if not f.exception():
            for r in f.result():
                q.put(r, timeout=5)

    def _thread_target():
        with futures.ProcessPoolExecutor(max_workers=4) as pool:
            fs = [pool.submit(_process, chunk) for chunk in _gen_chunks(xs, chunksize)]
            for f in fs:
                f.add_done_callback(_done_cb)
            done_and_not_done = futures.wait(fs)
            _raise_failed_or_not_done_futures(done_and_not_done)

    t = threading.Thread(target=_thread_target)
    t.start()
    return t


def _process(elems):
    time.sleep(1)
    return [e * 7 for e in elems]


def _raise_failed_or_not_done_futures(done_and_not_done):
    if done_and_not_done.not_done or any(f.exception() for f in done_and_not_done.done):
        exceptions = [f.exception() for f in done_and_not_done.done if f.exception()]
        raise RuntimeError(f"Exceptions: {exceptions}")


def gen_from_queue(q, num_results):
    for _ in range(num_results):
        try:
            x = q.get(timeout=30)
            yield x
        except queue.Empty:
            print("Gave up waiting")
            return


if __name__ == "__main__":
    # doesnt_work()
    # to_queue_with_vanilla_processes()

    xs = range(25)
    q = queue.Queue(maxsize=10)

    t = parallel_process_to_queue(xs, 2, q)

    for x in gen_from_queue(q, len(xs)):
        print(f"Received: {x}")

    t.join()
    # with futures.ThreadPoolExecutor(1) as pool:
    #     done_and_not_done
