import numpy as np
import random
from sklearn import preprocessing
import math
import sys
import matplotlib.pyplot as plt
import pylab

theta_poly = np.zeros((11, 1))
theta_line = np.zeros((6,1))
alpha = 0.5
ymax = 140.987
ymin = 103.38

data = []
with  open('/Users/zhaozehua/Desktop/train.txt', 'r') as fr:
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        data.append([(float(x))**2 for x in lineArr][:-1] + [float(x) for x in lineArr][:])
# normalization
min_max_scaler = preprocessing.MinMaxScaler()
data_minmax = min_max_scaler.fit_transform(data)
data_minmax = data_minmax.tolist()

# test dataset & training dataset
def dataset(data_minmax,dimensional):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for item in data_minmax:
        a = random.random()
        if a >= 0.2:
            if dimensional == 10:
                x_train.append(item[:dimensional])
            else:
                x_train.append(item[5:-1])
            y_train.append(item[-1])
        else:
            if dimensional == 10:
                x_test.append(item[:dimensional])
            else:
                x_test.append(item[5:-1])
            y_test.append(item[-1])
    x_train = np.array(x_train)
    x_train = x_train.reshape((-1, dimensional))
    y_train = np.array(y_train)
    y_train = y_train.reshape((-1, 1))
    yminarr1 = np.array([ymin] * y_train.shape[0])
    yminarr1 = yminarr1.reshape((-1, 1))
    y_train = y_train * (ymax - ymin) + yminarr1
    x_test = np.array(x_test)
    x_test = x_test.reshape((-1, dimensional))
    y_test = np.array(y_test)
    y_test = y_test.reshape((-1, 1))
    yminarr2 = np.array([ymin] * y_test.shape[0])
    yminarr2 = yminarr2.reshape((-1, 1))
    y_test = y_test * (ymax - ymin) + yminarr2
    # add one column for theta0
    x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
    return x_train,y_train,x_test,y_test

# gradient descent
def gradientDescent(X, y, theta, alpha, iters):
    m = y.shape[0]
    J_history = np.zeros((iters, 1))
    for i in range(iters):
        theta = theta - (alpha / m) * (X.T.dot(X.dot(theta) - y))
        C = X.dot(theta) - y
        J_history[i] = (C.T.dot(C)) / (2 * m)
    return J_history, theta

#linear regression
x_train_line,y_train_line,x_test_line,y_test_line = dataset(data_minmax,5)
J_history_line,theta_line = gradientDescent(x_train_line, y_train_line, theta_line, alpha, 1000)
C_line = x_test_line.dot(theta_line) - y_test_line
cost_line = (C_line.T.dot(C_line)) / (2 * y_test_line.shape[0])
print cost_line


#polynomial regression
x_train_poly,y_train_poly,x_test_poly,y_test_poly = dataset(data_minmax,10)
J_history_poly,theta_poly = gradientDescent(x_train_poly, y_train_poly ,theta_poly, alpha, 1000)
C_poly = x_test_poly.dot(theta_poly) - y_test_poly
cost_poly = (C_poly.T.dot(C_poly)) / (2 * y_test_poly.shape[0])
print cost_poly




plt.figure()
plt.plot(J_history_poly,label="polynomial",linewidth=0.5,)
plt.plot(J_history_line,label="linear",linewidth=0.5,)
plt.ylabel("sums of squared error",fontsize=15)
plt.xlabel("iterations",fontsize=15)
plt.legend()
plt.savefig("/Users/zhaozehua/Desktop/cost.eps")