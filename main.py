import csv
import random
from math import floor
from src.KNN import *
from src.logistic_regression import *

#open file
read = list(csv.reader(open("heart.csv", mode='r', encoding='utf-8-sig')))
column = read[:1]
data = read[1:]
random.shuffle(data)

#create dictionary from column name
columnIndex = {}
for i in range (len(column[0])):
    columnIndex[column[0][i]] = i

def driverKNN(n, K, target):
    #split to predict the first n entries as if unlabeled
    #ntar biar bisa liat akurasinya
    test = data[:n]
    train = data[n:]

    accuracy = 0
    for t in test:
        if(KNN(t, train, columnIndex[target], K)):
            accuracy += 1
        print()

    print("accuracy is",(float(accuracy)/n*100),"%")

def driverLogReg(lr, epoch, target):
    #set treshold
    treshold1 = 0.5

    #test 10 aja
    test = data[:10]
    train = data[10:]
    coef = train_logistic_regression(train, columnIndex[target], lr, epoch)

    accuracy = 0
    for i in test:
        print(log_reg_predict(coef, i, columnIndex[target]))
        if(log_reg_predict(coef, i, columnIndex[target]) >= treshold1):
            print("prediction :",1,"while actual value : ",i[13])
        else:
            print("prediction :",0,"while actual value : ",i[13])



#jalanin
# driverKNN(8, 5, 'target')
driverLogReg(0.01, 5, 'target')