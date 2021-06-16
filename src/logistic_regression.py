import math
from random import randint

def train_logistic_regression(train, labelIdx, lr, epochs):
    #set initial values for coefficients = 0
    #coefficients[0] is b0 and setting labelIdx to -1 bc it doesnt have a coefficient
    coefficients = [0.0 for i in range (len(train[0])+1)]
    coefficients[labelIdx] = -1

    #train n times, according to epochs value
    for i in range(epochs):
        #stochastic gradient descent, use 1 to measure loss, pick randomly
        stoc = train[randint(0,len(train)-1)]

        #fit and refine
        prediction = log_reg_predict(coefficients, stoc, labelIdx)
        refineCoefficient(coefficients, prediction, labelIdx)
    
    #function returns final coefficients
    return coefficients

def refineCoefficient(coeff, prediction, testEntry, labelIdx, lr):
    for i in range(len(coeff)):
        if(coeff[i] != -1):
            if(i == 0):
                coeff[i] += lr * (testEntry[labelIdx]-prediction) * prediction * (1 - prediction)
            else:
                coeff[i] += lr * (testEntry[labelIdx]-prediction) * prediction * (1 - prediction) * testEntry[i]

def log_reg_predict(coeff, entry, labelIdx):
    topower = coeff[0]
    for i in range (1,len(coeff)):
        if(coeff[i] != -1):
            topower += coeff[i]*float(entry[i])
    return 1/(1 + math.exp(-topower))