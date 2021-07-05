import numpy as np
from pandas.core.frame import DataFrame

#entropy dataset
def calcEntropy(dataFrem, target):
    entroSum = 0
    classes = dataFrem[target].unique()
    n = len(dataFrem[target])
    for kelas in classes:
        frac = dataFrem[target].value_counts()[kelas]/n
        entroSum += -frac*(np.log2(frac))
    return entroSum

#entropy relatif ke target
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
            entropyEachKelas += -fraction*np.log2(fraction+eps)
        fraction2 = den/len(dataFrem)
        classEntropy += -fraction2*entropyEachKelas
    

    return(abs(classEntropy))


def entropies(dataFream, target):
    halo = {}
    for i in list(dataFream.columns):
        halo[i] = calcEntropyAtr(dataFream, i, target)
    return halo

def attributeIGs(df, target):
    dataEnt = calcEntropy(df, target)
    entroDict = entropies(df, target)
    for i in list(df.columns):
        entroDict[i] = dataEnt - entroDict[i]
    return entroDict

def calcInfoGain():
    print("aaaaa")