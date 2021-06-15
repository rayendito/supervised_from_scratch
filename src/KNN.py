from math import sqrt

#calculate how far each entries are from each other using euclidean
def calculateEucDistance(row1, row2, labelIndex):
    dist = 0
    for i in range(len(row1)):
        if (i != labelIndex):
            dist += (float(row1[i])-float(row2[i]))**2
    return sqrt(dist)

def KNN(toPredict, train, labelIndex, K):
    neighbors = []
    for i in train:
        neighbors.append((round(calculateEucDistance(toPredict,i,labelIndex),2), i[labelIndex]))
    
    #sort based on the lowest value
    neighbors.sort(key=lambda x:x[0])

    #counting neighbor categories within K
    zero = 0
    one = 0
    for i in range(K):
        if(int(neighbors[i][1]) == 0):
            zero += 1
        else:
            one += 1
    
    #deciding
    if(zero > one):
        prediction = 0
    else:
        prediction = 1
    
    print("predicted result is",prediction)
    print("while it's actually",toPredict[labelIndex])
    print("nearest neighbors are",neighbors[:K])
    return prediction == int(toPredict[labelIndex])

