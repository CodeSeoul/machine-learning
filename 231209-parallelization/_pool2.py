import os 
import multiprocessing 

def twice(n):
    print(f"PID for {n}: {os.getpid()}")
    return (n*2) 
  
if __name__ == "__main__": 
    mylist = [1,2,3]

    p = multiprocessing.Pool() 
    res = p.map(twice, mylist)
  
    print(res)