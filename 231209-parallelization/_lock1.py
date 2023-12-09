from multiprocessing import Process, Value

def increment_counter(shared_counter):
    for _ in range(1000):
        shared_counter.value += 1

if __name__ == '__main__':
    shared_counter = Value('i', 0)
    processes = []

    for _ in range(4):
        process = Process(target=increment_counter, args=(shared_counter,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("Final Counter Value (without lock):", shared_counter.value)
