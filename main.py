import csv
import random
import pandas as pd
import numpy as np
from math import floor
from src.KNN import *
from src.logistic_regression import *
from src.id3 import *

#input kolom dan target yang bakal dimasukin
# UNCOMMENT WHEN DONE
# dataset = input("Masukkan nama file dataset : ")
# raw_string = r"{}".format(dataset)
# koloms = [item for item in input("Masukkan kolom2 atribut : ").split()]
# target = input("Masukkan kolom target : ")

# DELETE WHEN DONE
dataset = "heart.csv"
raw_string = r"{}".format(dataset)
koloms = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
target = 'target'

#open file using pandas and shuffle
df = pd.read_csv (raw_string)
df = df.sample(frac=1).reset_index(drop=True)

toProcess = df[koloms]
data = toProcess.values.tolist()
np.random.shuffle(data)
'''
*with target (0/1)

For categorical Attributes :
sex cp fbs restecg exang slope ca thal target

For non-categorical Attributes :
age trestbps chol thalach oldpeak target
'''

#create dictionary from column name
columnIndex = {}
column = list(toProcess.columns)
for i in range (len(column)):
    columnIndex[column[i]] = i

def driverKNN(n, K, target):
    #split to predict the first n entries as if unlabeled
    #not actually test and train, cuma biar ntar bisa liat akurasinya
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
    test = data[:8]
    train = data[8:]

    coef = train_logistic_regression(train, columnIndex[target], lr, epoch)

    accuracy = 0
    print("final coefficients:")
    print(coef)
    print()
    for i in test:
        pred = log_reg_predict(coef, i, columnIndex[target])
        if(pred >= treshold1):
            print("expected :",i[columnIndex[target]],"predicted :",pred,"[1]")
        else:
            print("expected :",i[columnIndex[target]],"predicted :",pred,"[0]")

def driverID3():
    a = buildTree(toProcess, target, [target])
    print(a)


#jalanin
# driverKNN(10, 5, target)
# driverLogReg(0.1, 150, target)
driverID3()

# dic = {'wkwkw' : {'aaa' : 1111,
#                     'bbb' : 2222}}
# dic['wkwkw']['cok'] = 3

# print(dic.get('wkwkw').get('cok'))