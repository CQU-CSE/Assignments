
"""
Created on Fri Dec 08 21:12:12 2017

@author: harry
"""

import pandas as pd
import numpy as np
trainSet=[]
testSet=[]
def dataSplit():
    dataSet=pd.read_csv('./mars_tianchi_user_actions.csv')
    dataSet=(np.array(dataSet))
    columns=['user_id','song_id','gmt_create','action_type','Ds']
    for item in dataSet:
        if item[4] in range(20150301,20150630):
            trainSet.append(item)
        else:
            testSet.append(item)
    pdTrainSet=pd.DataFrame(columns=columns,data=trainSet)
    pdTestSet=pd.DataFrame(columns=columns,data=testSet)
    pdTrainSet.to_csv('./trainset.csv',index=False,header=False)
    pdTestSet.to_csv('./testset.csv',index=False,header=False)
    print len(trainSet)
    print len(testSet)
   
if __name__ == '__main__':
    dataSplit()
         