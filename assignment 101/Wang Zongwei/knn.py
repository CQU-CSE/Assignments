#coding:utf8
import sys
import numpy as np
from array import *

answer_result = {}
train_result = {}
answer = []
trainSet = []
trainLabel = []         #labels of the samples in the training set
testSet = []
testLabel = []
predictions = []
K = 5                   #you can choose another K value
     
def kNN(newInput, dataSet, labels, k):
    traindatalength = dataSet.shape[0]#get the num of data

    diff = np.tile(newInput, (traindatalength, 1)) - dataSet

    squaredDiff = diff ** 2
    
    squaredDist = np.sum(squaredDiff, axis = 1)
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)
    
    classCount = {}
    for i in xrange(k): 
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    maxCount = 0  
    for key, value in classCount.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
  
    return maxIndex

def classify():
    with open('./training.txt') as f:
        content = f.readlines()
        for i in content:
            i = i.split(',')
            train=[]
            lable=[]
            train.append(float(i[0]))
            train.append(float(i[1]))
            train.append(float(i[2]))
            train.append(float(i[3]))
            if train_result.has_key(i[4].replace("\n","")):
                train_result[i[4].replace("\n","")]+=1
            else:
                train_result[i[4].replace("\n","")]=1
            trainLabel.append(i[4].replace("\n",""))
            trainSet.append(train)
            #trainLabel.append(lable)

    with open('./test.txt') as f:
        content = f.readlines()
        for i in content:
            i = i.split(',')
            test=[]
            test.append(float(i[0]))
            test.append(float(i[1]))
            test.append(float(i[2]))
            test.append(float(i[3]))
            #test.append(float(i[3].replace("\n","")))
            testLabel.append(i[4].replace("\n",""))
            testSet.append(test)
            
            

def getAccuracy(testLable, predictions):
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    Positive = 'Iris-virginica'
    for x in range(len(testSet)):
        if testLable[x] == Positive and predictions[x] == Positive:
            TP += 1
        if testLable[x] == Positive and predictions[x] != Positive:
            FN +=1
        if testLable[x] != Positive and predictions[x] != Positive:
            TN +=1
        if testLable[x] != Positive and predictions[x] == Positive:
            FP +=1
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = (2*TP)/(2*TP+FP+FN)

    return P, R, F1

classify()
trainSet = np.array(trainSet)
trainLabel = np.array(trainLabel)
testSet = np.array(testSet)

for i in range(len(testSet)):
    """
    print testSet[i]
    print trainSet
    print trainLabel
    """
    answer = kNN(testSet[i], trainSet, trainLabel, K)
    if answer_result.has_key(answer):
        answer_result[answer]+=1
    else:
        answer_result[answer]=1
    predictions.append(answer)
print train_result
print answer_result
print getAccuracy(testLabel,predictions)
    


    

