from multiprocessing import Process
res = [] 
  
def twice(mylist): 
    global res 
    for num in mylist: 
        res.append(num * 2) 
    print(f"res(in process p2): {res}") 
  
if __name__ == "__main__": 
    mylist = [1,2,3]
    # creating new process 
    p2 = Process(target=twice, args=(mylist,)) 
    # starting process 
    p2.start() 
    # wait until process is finished 
    p2.join() 

    print(f"res(in main program): {res}")