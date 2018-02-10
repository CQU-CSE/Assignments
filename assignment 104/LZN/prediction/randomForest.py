# coding=utf-8

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pandas as pd

base_dir = os.getcwd()
traindata = np.loadtxt(base_dir + r"\xtrain_new_del_ds_add_weekandfestival.csv", delimiter=",")
traindata_label = np.loadtxt(base_dir + r"\ytrain_new.csv", delimiter=",")
testdata = np.loadtxt(base_dir + r"\xtest_new_del_ds_add_weekandfestival.csv", delimiter=",")
testdata_label = np.loadtxt(base_dir + r"\ytest_new.csv", delimiter=",")

traindataLen = len(traindata)
testdataLen = len(testdata)

xtrainList = []  #样本数据集
ytrainList = []  #标签集

xtestList = []
ytestList = []

for i in range(traindataLen):
    trainrow = traindata[i]
    xtrainList.append(trainrow)
    trainrow_label = traindata_label[i]
    ytrainList.append(trainrow_label)

for j in range(testdataLen):
    testrow = testdata[j]
    xtestList.append(testrow)
    testrow_label = testdata_label[j]
    ytestList.append(testrow_label)

#把列表转成数组
xTrain = np.array(xtrainList)
yTrain = np.array(ytrainList)
xTest = np.array(xtestList)
yTest = np.array(ytestList)


mse = []

depth = None#建议设置：None
maxFeat = None  #最大属性值个数
RFModel = ensemble.RandomForestRegressor(n_estimators=35, max_depth=depth, max_features=maxFeat,random_state=0)
RFModel.fit(xTrain, yTrain)
prediction = RFModel.predict(xTest)
mse.append(mean_squared_error(yTest, prediction))
print mse


df = pd.read_csv('../dataset/testdata.csv', names=['user_id', 'song_id', 'Ds', 'play_num', 'download_num', 'collect_num',
                                                   'artist_id', 'publish_time', 'song_init_plays', 'Language','Gender'])
df['play_num']=prediction

df.to_csv('../dataset/prediction_del_ds_add_weekandfestival_feature.csv')


