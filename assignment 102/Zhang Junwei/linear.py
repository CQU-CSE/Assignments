# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:19:10 2017

@author: Administrator
"""

import numpy as np
testSet = []
trainSet = []

def Train():
    # ------------------- 特征缩放-------------------------
    with open('./noise_train.txt') as f:
        for line in f:
           #add your code here
           line = line.strip()
           trainlist = line.split(',')
           trainlist.insert(0, '1')
           trainSet.append(trainlist[0:7])
           #testLabel.append(testlist[-1])
    m = len(trainSet)
    n = len(trainSet[0])
    trainMat = np.array(np.ones((m,n)))
    for i in range(m):
        for j in range(n):
            trainMat[i][j] = trainSet[i][j]
    #print('1:', testMat)    # 测试1
    mean = []
    std = [] 
    for i in range(n-1):
        mean = np.mean(trainMat[:,i])
        std = np.std(trainMat[:,i])
        if std == 0:
            continue
        trainMat[:,i] = (trainMat[:,i]-mean)/std

    #print('2:', testMat)    # 测试2
    x = np.array(np.ones((m, n-1)))
    y = np.array(np.ones((m, 1)))
    for i in range(n-1):
        x[:,i] = trainMat[:,i]
    y = trainMat[:,n-1]
    
    return trainMat, x, y


def Test():
    # ------------------- 特征缩放-------------------------
    with open('./test.txt') as f:
        for line in f:
           #add your code here
           line = line.strip()
           testlist = line.split(',')
           testlist.insert(0, '1')
           testSet.append(testlist[0:7])
           #testLabel.append(testlist[-1])
    m = len(testSet)
    n = len(testSet[0])
    testMat = np.array(np.ones((m,n)))
    for i in range(m):
        for j in range(n):
            testMat[i][j] = testSet[i][j]
    #print('1:', testMat)    # 测试1  
    mean = []
    std = [] 
    for i in range(n-1):
        mean = np.mean(testMat[:,i])
        std = np.std(testMat[:,i])
        if std == 0:
            continue
        testMat[:,i] = (testMat[:,i]-mean)/std

    #print('2:', testMat)    # 测试2
    x = np.array(np.ones((m, n-1)))
    y = np.array(np.ones((m, 1)))
    for i in range(n-1):
        x[:,i] = testMat[:,i]
    y = testMat[:,n-1]
    return testMat, x, y


# ---------------------支持多变量的假设h函数-----------------------
# 输入：theta为1*m的参数向量 ,x为m*1的特征向量
# 输出：theta.dot(x)的浮点数
def h(theta, x):
    #print('fit')
    if theta is not None:
        return np.dot(theta, x)
    else:
        currenttheta = np.array(np.ones(1,len(x)))
        return np.dot(currenttheta, x)


# ----------------------损失函数 --------------------------------------
# J(θ0,...,θn)=(1/2m)*∑(hθ(xi)-yi)**2
# 输入：theta为1*m的参数向量 ，x为m*1的特征向量，y为
def cost(theta, x, y):
    sumOfSerie = 0
    for i in range(len(x)):
        #print((x[i].dot(theta) - y[i]))
        sumOfSerie += ((x[i].dot(theta) - y[i]) ** 2)
    print(sumOfSerie/(2*len(x)))
    return sumOfSerie/(2*len(x))
    

# ---------------------线性模型梯度下降 --------------------------------------
# θj := θj-αd(j)/d(θj)
# 拟合参数θ值，通过梯度下降，寻找最小的损失函数，从而确定最合适的θ值
# alpha为步长，
def gradientDescentTillConvergence(x, y, theta, alpha):
    numberOfx = len(x)   #行数
    previousCost = 0
    #print(x[0])

    a = np.array(np.ones( (len(x[0]),1 )))
    while True:
        for i in range (len(x[0])):
            sumdifgrad = 0
            for j in range(numberOfx):
                sumdifgrad = sumdifgrad+(x.dot(theta)[j] - y[j])*x[j,i]
            a[i][0] = (alpha/len(x))*sumdifgrad
        
        for k in range(len(x[0])):
            theta[k] = theta[k]-a[k]
            
        currentCost = cost(theta, x, y)

        difference = previousCost-currentCost
        if abs(difference)<0.000001:
            break
        previousCost = currentCost  
       # print(currentCost)
    #print i
    return theta
       

# --------------多项式回归模型----------------------------
def p(rate, l1, l2, x, y, degree):
    numberOfx = len(x)   #行数
    order = degree+1
    previousCost = 0
    #print(x[0])
    item = 0
    a = np.array(np.ones( (len(x[0]),1 )))
    features = np.array( np.empty([len(x), order], dtype=float) )
    theta = np.array( np.ones((order,1)) )
    
    for i in range(len(x)):
        for j in range(1,order):
            features[i][j] = pow(x[i][0], j) + pow(x[i][1], j) + pow(x[i][2], j) + pow(x[i][3], j) + pow(x[i][4], j) + pow(x[i][5], j)
            features[i][0] = 1
    while True:
        for i in range (len(features[0])):
            sumdifgrad = 0
            for j in range(numberOfx):
                sumdifgrad = sumdifgrad+(features.dot(theta)[j] - y[j])*features[j,i]
            a[i][0] = (rate/len(x))*sumdifgrad 
        
        for k in range(len(features[0])):
            theta[k] = theta[k]-a[k]
            
        currentCost = cost(theta, features, y) + l1 + l2*np.power(theta,2).sum()
        item = item+1
        difference = previousCost-currentCost
        if abs(difference)<0.00001:
            break
        previousCost = currentCost  
    print(item)
    return theta


# -------线性回归模型预测函数--------------
def liner_predict(theta, x):
    n = len(x)
    for i in range(n):
        print('the prediction is:',x[i].dot(theta))
    

# ------多项式回归模型预测函数--------------
def Polynomial_predict(theta, x, degree):
    order = degree+1
    features = np.array( np.empty([len(x), order], dtype=float) )
    for i in range(len(x)):
        for j in range(1,order):
            features[i][j] = pow(x[i][0], j) + pow(x[i][1], j) + pow(x[i][2], j) + pow(x[i][3], j) + pow(x[i][4], j) + pow(x[i][5], j)
            features[i][0] = 1
            
    for i in range(len(x)):
       print('the prediction is:',features[i].dot(theta))

    
trainMat, x, y = Train()
testMat, tx, ty = Test()
theta = np.array(np.ones((len(x[0]),1)))
#cost = cost(theta, x, y)
#x.dot(theta)  #n行1列
alpha = 0.1
l1 = 0.001
l2 = 0.001
degree = 2
#currenttheta = gradientDescentTillConvergence(x, y, theta, alpha)
curtheta = p(0.01, l1, l2, x, y, degree)

#liner_predict(currenttheta, testMat)
#Polynomial_predict(curtheta, tx, degree)
    