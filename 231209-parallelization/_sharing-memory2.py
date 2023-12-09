import multiprocessing
from multiprocessing import Process, Value, Array 
  
def twice(mylist, res): 
    for idx, num in enumerate(mylist): 
        res[idx] = num * 2
    
    print(f"res(in process p2): {res[:]}") 
  
if __name__ == "__main__": 
    mylist = [1,2,3]
    # creating Array of int data type with space of 3
    res = multiprocessing.Array('i', 3) 
    # creating new process 
    p2 = multiprocessing.Process(target=twice, args=(mylist, res)) 
    # starting process 
    p2.start() 
    # wait until the process is finished 
    p2.join() 
  
    # print result array 
    print(f"Result(in main program): {res[:]}") 