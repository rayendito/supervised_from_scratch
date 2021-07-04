import numpy as np

def calcEntropy(dataFrem, target):
    entroSum = 0
    classes = dataFrem[target].unique()
    n = len(dataFrem[target])
    for kelas in classes:
        frac = dataFrem[target].value_counts()[kelas]/n
        entroSum += -frac*(np.log2(frac))
    return entroSum

def calcEntropyAtr(dataFrem, attribute, target):
    # in case pembaginya 0, eps angka kecil tidak nol
    eps = np.finfo(float).eps

    goals = dataFrem[target].unique()
    classes = dataFrem[attribute].unique()

    classEntropy = 0
    for kelas in classes:
        entropyEachKelas = 0
        den = len(dataFrem[attribute][dataFrem[attribute] == kelas])
        for goal in goals:
            num = len(dataFrem[attribute][dataFrem[attribute] == kelas][dataFrem[target] == goal])
            fraction = num/(den+eps)
            entropyEachKelas += -fraction*np.log2(fraction+eps) #This calculates entropy for one feature like 'Sweet'
        fraction2 = den/len(dataFrem)
        classEntropy += -fraction2*entropyEachKelas
    
    return(abs(classEntropy))


def calcInfoGain():
    print("aaaaa")