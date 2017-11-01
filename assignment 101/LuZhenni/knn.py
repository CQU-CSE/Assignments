# coding:utf8
import sys
import numpy as np
import math
import operator


K = 5 # you can choose another K value'''

def calDistance(point1, point2, length):#计算两点之间的欧式距离
    distance=0
    for x in range(length):#这里的length是指实例有几项，对应于特征向量的维度
        distance += pow((point1[x]-point2[x]),2)
    return math.sqrt(distance)

def getCategory(trainingSet,testPoint,K):#获取离样本点最近的k个训练实例点
    neighborList = []#离样本点最近的k个训练实例点的列表
    distanceList=[]#由训练实例及其与样本点的欧氏距离组成的列表
    length=len(testPoint)-1#－1，因为测试集里面是有类标签的
    categories = []

    for x in range(len(trainingSet)):
        dist=calDistance(testPoint,trainingSet[x],length)
        distanceList.append([trainingSet[x],dist])
    distanceList.sort(key=operator.itemgetter(1))
    for x in range(K):
        neighborList.append(distanceList[x][0])#存储的是前k个训练实例点的distance列表中的(trainingSet[x])
    neighborsLabel0=dict.fromkeys(['Iris-setosa\n','Iris-virginica\n','Iris-versicolor\n'],0)#字典中对应的键值对是类别及数量

    for x in range(len(neighborList)):
        categories.append(neighborList[x][-1])#训练实例点的最后一个属性，即类标记
        for i in categories:
            if i =='Iris-setosa\n':
                neighborsLabel0[i]+=1
            if i =='Iris-virginica\n':
                neighborsLabel0[i] += 1
            else:
                neighborsLabel0[i] += 1
    neighborsLabel1=sorted(neighborsLabel0.iteritems(),key=operator.itemgetter(1),reverse=True)#将字典以迭代器对象返回，并按类的数量进行降序排序
    return neighborsLabel1[0][0]

def calAccuracy(testSet,answer):#计算准确度
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
        classifyResult=getCategory(trainingSet,testSet[x],K)
        answer.append(classifyResult)
        print'testData:'+(repr(testSet[x])+'\t\tpredicted:'+repr(classifyResult))
    accuracy=calAccuracy(testSet,answer)
    print ('Accuracy:'+repr(accuracy)+'%')

if __name__ == '__main__':
    classify()
