# coding:utf8
import sys
import numpy as np
import math
import operator


K = 5 # you can choose another K value'''

def euclideanDistance(instance1, instance2, length):#计算两点之间的欧式距离
    distance=0
    for x in range(length):#这里的length是指实例有几项，对应于特征向量的维度
        distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

def getNeighbors(trainingSet,testInstance,K):#获取离样本点最近的k个训练实例点
    neighbors = []#离样本点最近的k个训练实例点的列表
    distance=[]#由训练实例及其与样本点的欧氏距离组成的列表
    length=len(testInstance)-1#－1，因为测试集里面是有类标签的
    for x in range(len(trainingSet)):
        dist=euclideanDistance(testInstance,trainingSet[x],length)
        distance.append([trainingSet[x],dist])
    distance.sort(key=operator.itemgetter(1))
    for x in range(K):
        neighbors.append(distance[x][0])#存储的是前k个训练实例点的distance列表中的(trainingSet[x])
    return neighbors

def getResponse(neighbors):#获取k个训练实例点的最多的类
    neighborsLabel={}#字典中对应的键值对是类别及数量
    for x in range(len(neighbors)):
        response=neighbors[x][-1]#训练实例点的最后一个属性，即类标记
        if response in neighborsLabel:#遍历k个训练实例点的response，如果在字典中，就加1，不在的话更新这个值为1
            neighborsLabel[response]+=1
        else:
            neighborsLabel[response]=1
    sortedVotes=sorted(neighborsLabel.iteritems(),key=operator.itemgetter(1),reverse=True)#将字典以迭代器对象返回，并按类的数量进行降序排序
    return sortedVotes[0][0]#返回列表里第一项的类别

def getAccuracy(testSet,answer):#计算准确度
    correct=0#表示测试集中预测正确的个数
    for x in range(len(testSet)):
        if testSet[x][-1] == answer[x]:#如果预测的类标签与实际的类标签相同，则预测正确的个数加1
            correct += 1
    return (correct/float(len(testSet)))*100.0

def classify():
    # training
    trainingSet=[]
    with open('./training.txt') as f:
        for line in f:
            trainingSet.append(line.split(','))
    for x in range(len(trainingSet)):
        for y in range(4):
            trainingSet[x][y]= float(trainingSet[x][y])
    # you can add your core code here

    # test
    testSet=[]
    with open('./test.txt') as f:
        for line in f:
            testSet.append(line.split(','))
    for x in range(len(testSet)):
        for y in range(4):
            testSet[x][y] = float(testSet[x][y])
    # add your code here

    answer=[]
    for x in range(len(testSet)):
        neighbors=getNeighbors(trainingSet,testSet[x],K)
        result=getResponse(neighbors)
        answer.append(result)
        print(repr(testSet[x])+'>predicted='+repr(result))
    accuracy=getAccuracy(testSet,answer)
    print ('Accuracy:'+repr(accuracy)+'%')
if __name__ == '__main__':
    classify()
