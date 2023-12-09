from multiprocessing import Process, Manager

def _addmember(d):
    d["members"]+=1
    print(f"Welcome to {d['name']}, we are {d['members']} now")

if __name__ == '__main__':
    manager = Manager()
    group = manager.dict()
    group["name"] = "CodeSeoul"
    group["members"] = 100
    p2 = Process(target=_addmember,args=(group,))
    p2.start()
    p2.join()

    print(f"Result(in main program): {group['members']}")