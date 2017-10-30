#Create by keagan
#Date:10.25
import csv
import random
import math
import operator

#read txt file as csv
#munb means that txt have mumb columns
def loadDataset(filename, split, munb,trainingSet):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        #to split the line to munb columns
        for x in range(len(dataset)-1):
            for y in range(munb):
                dataset[x][y] = float(dataset[x][y])
            trainingSet.append(dataset[x])

#use formula to calculate the distance of two vectors
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):#the dimension of the eigenvector
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)

#to determine the nearest K nodes
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

#if konw the actual classification in txt file use that to calculate numbs;
"""
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0
"""



def main():
    #prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset(r'E:/assignment/training.txt', split,4,trainingSet)
    loadDataset(r'E:/assignment/test.txt', split,3, testSet)
    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))
    #generate predictions
    predictions = []
    k = 5
    for x in range(len(testSet)):
        # trainingsettrainingSet[x]
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print ('>predicted=' + repr(result)+',')
        #to print the predicted and actual classification
        #print ('>predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    #use that to calculate the correct rate
    #print ('predictions: ' + repr(predictions))
    #accuracy = getAccuracy(testSet, predictions)
    #print ('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()