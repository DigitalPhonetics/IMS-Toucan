if __name__ == '__main__':
    l = [0, 1, 2, 3]
    for i, cont in enumerate(l):
        print(i, cont)
        if i < len(l) - 1:
            print("yes")
        else:
            print("no")
