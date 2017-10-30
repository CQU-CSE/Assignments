#coding:utf8
import sys
import numpy as np
from collections import Counter  
answer = []
trainSet = []
trainLabel = []         #labels of the samples in the training set
K = 5          #you can choose another K value
distList=[]             #记录欧式距离
testSet=[]              #测试集
testLabel=[]            #用于存储测试标签预测值
testTrueLabel=[]        #存放测试集标签真实值
count=0                 #记录近邻点在训练集中的位置
TP=0


def classify():
    #training
    with open('./training.txt') as f:
        for line in f:            
            coords1=map(float,(line.split(','))[0:4])
            label=(line.split(','))[4]
            global trainSet
            global trainLabel
            trainSet.append(coords1)
            trainLabel.append(label)
            #print coords1
           #you can add your core code here

    #test
    with open('./test.txt') as f:
        for line in f:
            coords2=map(float,(line.split(','))[:])
            testSet.append(coords2)
            #print coords2
           #add your code here
        for i in range(0,len(testSet)):
            global distList
            distList=[]
            for j in range(0,len(trainSet)):
                global count            
                np_c1 = np.array(trainSet[j])  #列表转向量，以应用numpy计算欧氏距离
                np_c2 = np.array(testSet[i])
                distNow=eucldist_vectorized(np_c1,np_c2) #计算当前两个样本的距离
                count=j #记录距离最近的点
                distRecord=[distNow,count] #将距离与标记相关起来,放入列表中
                distList.append(distRecord)#添加到距离列表中
            distList.sort(key=lambda List:List[0]) #按距离升序排序
            shortestDist=distList[0:K] #选出距离最小的五个点,放入shortestDist中
            #选出了5个点，再实现多数表决算法（用counter）
            #print shortestDist输出5个近邻的距离以及它们在训练集中的编号
            Label=[] #创建一个列表存取距离最短的5个点的标签
            for k in range(0,K):
                Label.append((trainLabel[((shortestDist[k])[1])]))#存取距离最短点的标签                
            #print Label输出五个近邻的标签            
            C=Counter(Label)         #用counter求出在近邻点集中各类标签出现的次数，出现最多的即作为标签
            if C['Iris-setosa\n']>C['Iris-virginica\n'] and C['Iris-setosa\n']>C['Iris-versicolor\n']:
                testLabel.append(['Iris-setosa\n'])#将预测结果放入测试集标签集合中
            elif C['Iris-virginica\n']>C['Iris-setosa\n'] and C['Iris-virginica\n']>C['Iris-versicolor\n']:
                testLabel.append(['Iris-virginica\n'])
            else:
                testLabel.append(['Iris-versicolor\n'])
    with open('./testLabel.txt') as f:
        for line in f:
             testTrueLabel.append(line)
        #print testTrueLabel
        
# Calculates the euclidean distance between 2 lists of coordinates.计算两个坐标向量之间的欧氏距离
def eucldist_vectorized(coords1, coords2):
    return np.sqrt(np.sum((coords1 - coords2)**2))    #**表示次方
               

if __name__ == '__main__':

    classify()
    for i in range(0,len(testSet)):
        if testTrueLabel[i]==(testLabel[i])[0]:
            TP+=1
    correctness=float(TP)/float((len(testSet)))
   
    print correctness
#print trainSet
#print trainLabel
#print testLabel