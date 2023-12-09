import os, time, multiprocessing

def worker1():
    while True:
        time.sleep(2)
        print(f"Process PID {os.getpid()} is running")

def worker2():
    while True:
        time.sleep(2)
        print(f"Process PID {os.getpid()} is running")

if __name__ == "__main__":
    print("ID of main process: {}".format(os.getpid()))

    # creating processes
    p1 = multiprocessing.Process(target=worker1)
    p2 = multiprocessing.Process(target=worker2)

    # starting processes
    p1.start()
    p2.start()

    # process IDs
    print(f"Process PID for worker1: {p1.pid}")
    print(f"Process PID for worker1: {p2.pid}")