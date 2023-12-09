def twice(n):
    return (n*2)


if __name__ == "__main__":
    mylist = [1, 2, 3]

    res = []
    for num in mylist:
        res.append(twice(num))

    print(res)