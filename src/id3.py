def calcEntropy(dataFrem, target):
    entroSum = 0
    classes = dataFrem[target].unique()
    n = len(dataFrem[target])
    for kelas in classes:
        frac = dataFrem[target].value_counts[kelas]/n
        entroSum += -frac*np.log2(fraction)
    return entroSum

def calcInfoGain():
    print("aaaaa")