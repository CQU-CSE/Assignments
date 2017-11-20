"""
Create by Keagan Wang
Date 20/11
"""
import numpy as np
#To
np.set_printoptions(suppress=True)

def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()
    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = np.dot(xTrains, loss) /(2*m)
        theta = theta - alpha * gradient
    return theta

def polynomial(data):
    data1=data[:,0:1]**0
    data2=data[:, 1:2] **1
    data3=data[:, 2:3] **2
    data4=data[:, 3:4] **3
    data5=data[:, 4:5] **4
    data6 = data[:, 5:6] ** 5
    data=np.concatenate((data1,data2),axis=1)
    data=np.concatenate((data,data3),axis=1)
    data = np.concatenate((data, data4),axis=1)
    data = np.concatenate((data, data5),axis=1)
    data = np.concatenate((data, data6), axis=1)
    return data



def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n+1))
    xTest[:, :-1] = x
    yP = np.dot(xTest, theta)
    return yP


def main():
    trainingSet=np.loadtxt(open(r'/home/keagan/Documents/Python/asign/noise_train.txt'),delimiter=',')
    data=trainingSet[:,1:len(trainingSet)]
    trainLabel=trainingSet[:,-1]
    tempdata=np.ones((len(trainingSet),1))
    trainData=np.concatenate((tempdata,data),axis=1)
    print(trainData.shape)
    #多项式预测
    # trainData=polynomial(trainData)
    print(trainData.shape)
    m,n=trainData.shape
    theta = np.array([0.5,0.001,1,1,0.01,10])
    alpha = 0.0000001
    maxIteration =500
    theta = batchGradientDescent(trainData, trainLabel, theta, alpha,m, maxIteration)
    testDataSet = np.loadtxt(open(r'/home/keagan/Documents/Python/asign/noise_test.txt'), delimiter=',')
    print (predict(testDataSet, theta))

if __name__=='__main__':
    main()