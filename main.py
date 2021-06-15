import csv
from math import floor
from src.KNN import *

#open file
read = list(csv.reader(open("heart.csv", mode='r', encoding='utf-8-sig')))
column = read[:1]
data = read[1:]

#how many to be tested and K value
n = 8
K = 5

#create dictionary from column name
columnIndex = {}
for i in range (len(column[0])):
    columnIndex[column[0][i]] = i

#split to predict the first 5 entries as if unlabeled
#ntar biar bisa liat akurasinya
test = data[:n]
train = data[n:]

accuracy = 0
for t in test:
    if(KNN(t, train, columnIndex['target'], K)):
        accuracy += 1
    print()

print("accuracy is",(float(accuracy)/n*100),"%")

# print(data[0])
# print(columnIndex)
# print(arraydata[0])
