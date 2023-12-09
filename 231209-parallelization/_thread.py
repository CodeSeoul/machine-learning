import threading

# Define a simple function that will be executed in a thread
def print_numbers():
    for i in range(5):
        print(i)

if __name__ == "__main__":
    # Create a Thread object and target it to the function
    my_thread = threading.Thread(target=print_numbers)

    # Start the thread
    my_thread.start()
    # Wait for the thread to finish (optional)
    my_thread.join()

    # Continue with the main program
    print("Main program continues...")