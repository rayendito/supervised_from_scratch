from math import sqrt

#calculate how far each entries are from each other using euclidean
def calculateEucDistance(row1, row2, labelIndex):
    dist = 0
    for i in range(len(row1)):
        if (i != labelIndex):
            dist += (row1[i]-row2[i])**2
    return sqrt(dist)

def KNN(existingSet, toPredict, labelIndex, K):
    neighbors = []
    for row in existingSet:
        