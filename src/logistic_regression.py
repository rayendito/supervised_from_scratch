from math import exp, sqrt
from random import randint

#it turns out that i haven't rescaled the attributes yet so it keeps getting Overflow error :'D
##RESCALING##
def minmaxValue(train):
    minmax = []
    for i in range (len(train[0])):
        kolom = [(kol[i]) for kol in train]
        minmax.append([min(kolom), max(kolom)])
    return minmax

def rescale(train, minmax):
    pnjng = len(train[0])
    for row in train:
        for i in range (pnjng):
            try:
                row[i] = ((row[i])-minmax[i][0])/(minmax[i][1]-minmax[i][0])
            except:
                print(minmax[i][1],minmax[i][0])

#REFINING COEFFICIENTS##
def refineCoefficient(coeff, prediction, testEntry, labelIdx, lr):
    coeff[0] += lr * ((testEntry[labelIdx])-prediction) * prediction * (1 - prediction)
    for i in range(1, len(coeff)):
        if(i != labelIdx+1):
            coeff[i] += lr * ((testEntry[labelIdx])-prediction) * prediction * (1 - prediction) * (testEntry[i-1])

def log_reg_predict(coeff, entry, labelIdx):
    topower = coeff[0]
    for i in range (1,len(coeff)):
        if(i != labelIdx+1):
            topower += coeff[i]*(entry[i-1])
    return (1/(1 + exp(-topower)))

##COUNTING LOSS##
def printLoss(predicted, actual):
    print("loss is :",round(sqrt((predicted-actual)**2),4))

##main?##
def train_logistic_regression(train, labelIdx, lr, epochs):
    #set initial values for coefficients = 0
    #coefficients[0] is b0
    coefficients = [0 for i in range (len(train[0])+1)]

    #rescaling properties
    minmax = minmaxValue(train)
    rescale(train, minmax)

    #train n times, according to epochs value
    for i in range(epochs):
        #stochastic gradient descent, use 1 to measure loss
        stoc = train[randint(0,len(train)-1)]

        #fit and refine
        prediction = log_reg_predict(coefficients, stoc, labelIdx)
        refineCoefficient(coefficients, prediction, stoc, labelIdx, lr)

        #print 
        print("Prediction :",prediction)
        print("Coefficients :",coefficients)
        printLoss(prediction, stoc[labelIdx])
        print()
    
    #function returns final coefficients
    return coefficients