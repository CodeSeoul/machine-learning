from multiprocessing import Process, Value, Lock

def _increment(shared_counter, lock):
    for _ in range(1000):
        with lock:
            shared_counter.value += 1

if __name__ == '__main__':
    shared_counter = Value('i', 0)
    lock = Lock()
    processes = []
    for _ in range(4):
        process = Process(
            target=_increment, args=(shared_counter, lock))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("Final Counter Value:", shared_counter.value)