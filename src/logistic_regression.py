from math import exp
from random import randint

#it turns out that i haven't rescaled the attributes yet so it keeps getting Overflow error :'D
def minmaxValue(train):
    minmax = []
    for i in range (len(train[0])):
        kolom = [kol[i] for kol in train]
        minmax.append([float(min(kolom)), float(max(kolom))])
    return minmax

def rescale(train, minmax):
    pnjng = len(train[0])
    for row in train:
        for i in range (pnjng):
            row[i] = (float(row[i])-minmax[i][0])/(minmax[i][1]-minmax[i][0])

def train_logistic_regression(train, labelIdx, lr, epochs):
    #set initial values for coefficients = 0
    #coefficients[0] is b0 and setting labelIdx to -1 bc it doesnt have a coefficient
    coefficients = [0 for i in range (len(train[0])+1)]
    coefficients[labelIdx+1] = -1

    #rescaling properties
    minmax = minmaxValue(train)
    rescale(train, minmax)

    #train n times, according to epochs value
    for i in range(epochs):
        #stochastic gradient descent, use 1 to measure loss, pick randomly (also rescale)
        stoc = [train[randint(0,len(train)-1)]]
        rescale(stoc, minmax)

        #fit and refine
        prediction = log_reg_predict(coefficients, stoc[0], labelIdx)
        print(prediction)
        refineCoefficient(coefficients, prediction, stoc[0], labelIdx, lr)

        #print coefficients
        for i in range(len(coefficients)):
            coefficients[i] = round(coefficients[i], 4)
        print("coefficients are now :")
        print(coefficients)
    
    #function returns final coefficients
    return coefficients

def refineCoefficient(coeff, prediction, testEntry, labelIdx, lr):
    for i in range(len(coeff)):
        if(i != labelIdx+1):
            if(i == 0):
                coeff[i] += lr * (float(testEntry[labelIdx])-prediction) * prediction * (1 - prediction)
            else:
                coeff[i] += lr * (float(testEntry[labelIdx])-prediction) * prediction * (1 - prediction) * float(testEntry[i-1])

def log_reg_predict(coeff, entry, labelIdx):
    topower = coeff[0]
    for i in range (1,len(coeff)):
        if(i != labelIdx+1):
            topower += coeff[i]*float(entry[i-1])
    try:
        return float(1/(1 + exp(-topower)))
    except OverflowError:
        print("Overflow gan idk")
        print(topower)