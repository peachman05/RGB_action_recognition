
def readfile_to_dict(filename):
    d = {}
    f = open(filename)
    for line in f:
        (key, val) = line.split()
        d[key] = int(val)

    return d