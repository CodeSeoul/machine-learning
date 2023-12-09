import multiprocessing

# Define a simple function that will be executed in a process
def print_numbers():
    for i in range(5):
        print(i)

if __name__ == "__main__":
    # Create a Process object and target it to the function
    my_process = multiprocessing.Process(target=print_numbers)

    # Start the process
    my_process.start()
    # Wait for the process to finish (optional)
    my_process.join()

    # Continue with the main program
    print("Main program continues...")