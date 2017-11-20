#coding:utf8
import random
import sys
import numpy as np
from sklearn.metrics import classification_report
import math

answer = []
trainSet = []
testSet = []
labeldata = []
"""
def testing():
    #test
    with open('./ml/test.txt') as f:
        for line in f:
           #you can add your code below
"""


def sigmoid(inX):
    return 1/(1+math.exp(-inX))

#随机梯度下降算法
def BGD(x, y,  alpha, maxIterations):
    m, n = x.shape
    theta = [120,0.001,1,1,0.01,10]
    theta = np.array(theta)
    #theta = np.full(n,0.001)
    for i in range(maxIterations):
        
        for k in range(m):
            #产生随机索引
            #这种方式好像更好理解一些
            """
            但是这种方法是错的
            randIndex = int(random.uniform(0, m))
            h = sigmoid(x[randIndex].dot(theta))
            gra = alpha*(sigmoid(y[randIndex])-h)*h
            theta = theta + gra
            
            """
            randIndex = int(random.uniform(0, m))
            h = (x[randIndex].dot(theta))
            #计算错误率
            error = h - y[randIndex]
            chazhi = alpha * (error * x[randIndex])
            

            #更新参数
            theta = theta - chazhi
            
    return theta

def cost(theta, x, y):
    sumOfSerie = 0
    for i in range(len(x)):
        sumOfSerie += ((x.dot(theta)[i] - y[i]) ** 2)
    return sumOfSerie/(2*len(x))

def PNM(x, y, degree):
    features = np.array( np.empty([len(x), degree+1], dtype=float) )
    theta = np.array( np.ones((degree+1,1)) )
    cost_prev = 0
    chazhi = np.array(np.ones((len(x), 1)))
    
    for i in range(len(x)):
        for j in range(degree+1):
            features[i][j] = pow(x[i][0], j) + pow(x[i][1], j) + pow(x[i][2], j) + pow(x[i][3], j) + pow(x[i][4], j) + pow(x[i][5], j) 

    log = 0        
    for repeat in range(500):
        for i in range(len(x)):
            chazhi[i] = features.dot(theta)[i]-y[i]
        theta[0] -= (0.000001/len(x))*(chazhi*features[:,0]).sum()
        for j in range(1, degree+1):
            theta[j] -= (0.000001/len(x))*(chazhi*features[:,j]).sum()
        # calculate cost function after every 10 iterations
        if(repeat%20 == 0):
            cost1 = cost(theta, features, y) + 0.001*theta.sum() + 0.001*np.sqrt(np.power(theta,2).sum())
            # if change in cost function < tolerance, break the loop
            if(log>1 and np.abs(cost1-cost_prev) < 0.00001):
                break
            log += 1
            cost_prev = cost1

    return cost_prev[0], theta

def readforlinear():
    with open('./noise_train.txt') as f:
        content = f.readlines()
        for i in content:
            i = i.split(',')
            traindata=[]
            traindata.append(1.0)
            traindata.append(float(i[0]))
            traindata.append(float(i[1]))
            traindata.append(float(i[2]))
            traindata.append(float(i[3]))
            traindata.append(float(i[4]))
            labeldata.append(float(i[5]))
            trainSet.append(traindata)

def readforpnm():
    with open('./noise_train.txt') as f:
        content = f.readlines()
        for i in content:
            i = i.split(',')
            traindata=[]
            traindata.append(1.0)
            traindata.append(float(i[0]))
            traindata.append(float(i[1]))
            traindata.append(float(i[2]))
            traindata.append(float(i[3]))
            traindata.append(float(i[4]))
            traindata.append(float(i[5]))
            trainSet.append(traindata)
#这个是用来线性回归的

readforlinear()
x = np.array(trainSet)
y = np.array(labeldata)

m, n = np.shape(x)
loopmax = 100
alpha = 0.00000000001
theta = BGD(x,y,alpha,loopmax)
print theta




output = 0
for i in range(len(x)):
    result= 0
    for j in range(len(x[i])):
        
        result += x[i][j] * theta[j]
    output += (result - y[i])**2
#print result
print '线性:',output/(2*len(x))



#这个是用来多项式回归的
"""
readforpnm()
m = len(trainSet)
n = len(trainSet[0])
trainMat = np.array(np.ones((m,n)))
for i in range(m):
    for j in range(n):
        trainMat[i][j] = trainSet[i][j]
mean = []
std = [] 
for i in range(n):
    mean = np.mean(trainMat[:,i])
    std = np.std(trainMat[:,i])
    if std == 0:
        continue
    trainMat[:,i] = (trainMat[:,i]-mean)/std

x = np.array(np.ones((m, n-1)))
y = np.array(np.ones((m, 1)))
for i in range(n-1):
    x[:,i] = trainMat[:,i]
y = trainMat[:,n-1]
degree = 2
print PNM(x,y,degree)
"""
