#coding:utf8
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib

answer = []
trainSet = []
trainLabel = []         #labels of the samples in the training set
K = 5                   #you can choose another K value
testSet = []
testLabel = []
predictions = []

## 归一化数据,保证特征等权重
## 返回规范的数据矩阵，最大值和最小值之间的范围，数据中的最小值
def autoNorm(dataSet):
    #dataSet.sort()
    normDataSet = []
    count = len(dataSet)
    returnMat = np.zeros((count,4))
    for x in range(count):
        for y in range(4):
            returnMat[x][y] = dataSet[x][y]
   
    # 求出最小值
    minVals = returnMat.min(0)
    maxVals = returnMat.max(0)
    ranges = maxVals - minVals

    normDataSet = np.zeros(np.shape(returnMat))##建立与dataSet结构一样的矩阵，并初始化为0
    m = returnMat.shape[0]#矩阵的行数
    for i in range(1,m):
        normDataSet[i,:] = (returnMat[i,:] - minVals) / ranges

    return normDataSet,ranges,minVals

##KNN算法
# 输入 单行测试数据集，全部训练集，全部训练集标签，K值
# 输出 分类结果
def knn(input,dataSet,label,k):
    dataSize = dataSet.shape[0]
    ####计算欧式距离
    #print("input:",input)
    #print("dataSet:",dataSet)
    diff = np.tile(input,(dataSize,1)) - dataSet   #构建与训练数据集大小相同矩阵，进行计算
    #print("diff:",diff)
    sqdiff = diff ** 2
    squareDist = np.sum(sqdiff,axis = 1)###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5
    
    ##对距离进行排序
    sortedDistIndex = np.argsort(dist)##argsort()根据元素的值从大到小对元素进行排序，返回下标

    #计算训练集中每类类别的数量
    num_label={}                 # 定义一个词典
    for i in range(len(testSet)):
        count_label = label[sortedDistIndex[i]]
        num_label[count_label] = num_label.get(count_label,0)+1
        
    # 计算K个训练集中每类类别的数量-【词典】
    classCount={}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
        
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key,value in classCount.items():
        #print(classCount[key])
        if value > maxCount:
            #print(value, key)
            maxCount = value
            classes = key
        if value == maxCount:      #如果存在类别相等的情况，判断相等类别中。在所有训练集中数目最多的一类
            if num_label[key]>=num_label[classes]:
                maxCount = value
                classes = key
    return classes

# 精确率函数
# 输入 测试数据集标签  预测数据集标签 正类值(关注的类值)
# 输出 准确率 精确率 召回率 F1值(精确率和召回率的调和均值)
def getAccuracy(testLable, predictions, Positive):
    TP = 0          #将正类预测为正类数
    FN = 0          #将正类预测为负类数
    TN = 0          #将负类预测为正类数
    FP = 0          #将负类预测为负类数
    correct = 0     #预测正确数据个数
    for x in range(len(testSet)):
        #print(testLable[x],"111",predictions[x], "222",Positive)
        if testLable[x] == predictions[x]:
            correct +=1
        if testLable[x] == Positive and predictions[x] == Positive:
            TP += 1
        if testLable[x] == Positive and predictions[x] != Positive:
            FN +=1
        if testLable[x] != Positive and predictions[x] != Positive:
            TN +=1
        if testLable[x] != Positive and predictions[x] == Positive:
            FP +=1
    #print("correct:",correct,"TP:",TP, "FN:",FN, "TN:",TN, "FP:",FP)

    accuracy = (correct/float(len(testSet)))*100.0
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = (2*TP)/(2*TP+FP+FN)

    return accuracy, P, R, F1

def classify():
    #training
    with open('./training.txt') as f:
        for line in f:
            #you can add your core code here
            line = line.strip()
            listFromLine = line.split(',')   #按照逗号分隔
            trainSet.append(listFromLine[0:4])
            trainLabel.append(listFromLine[-1])
        #print(trainSet)
        ##将列表的最后一列由字符串转化为数字，便于以后的计算
        dictClassLabel = Counter(trainLabel)
        classLabel = []
        kind = list(dictClassLabel)
        for item in trainLabel:
            if item == kind[0]:
                item = 1 
            elif item == kind[1]:
                item = 2
            else:
                item = 3
            classLabel.append(item)
        normMat,ranges,minVals = autoNorm(trainSet)
       
    #test
    with open('./test.txt') as f:
        for line in f:
           #add your code here
           line = line.strip()
           testlist = line.split(',')
           testSet.append(testlist[0:4])
           testLabel.append(testlist[-1])
        #测试数据归一化   
        testMat,ranges_test,minVals_test = autoNorm(testSet)
        #print("tesMat",testMat)
        m = normMat.shape[0]  #trainSet数据数量 
        testNum = testMat.shape[0]  #测试训练集的行数

        #print(normMat)
        for i in range(0,testNum):
            #测试数据集[i]标签=knn(册数数据集第i行所有数据，训练数据集从0到m行所有数据，训练数据集从0行到m行所有标签，K值)
            classifyResult = knn(testMat[i,:],normMat[0:m,:],trainLabel[0:m],K)
            predictions.append(classifyResult)
            #print("normMat",normMat[0:testNum,:])
            #print("tesMat",testMat[i,:])
            print("分类后的结果为:,", classifyResult)
        #计算准确率，精确率，召回率，F1值
        accuracy,P,R,F1 = getAccuracy(testLabel, predictions, 'Iris-virginica')
        print("准确率accuracy:",accuracy, "精确率P:",P,"召回率R：",R, "F1值:",F1)
	
if __name__ == '__main__':
    classify()

    

