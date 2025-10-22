def read_class_txt(filename):
    with open(filename, "r") as f:
        mydata = [line.strip() for line in f]

    return sorted(mydata)
