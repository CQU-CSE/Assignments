import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import math
from sklearn import preprocessing  
from sklearn.linear_model import Lasso
#use randomforest to predict

trainingSet = pd.read_csv('./finalTrainSetFirstTrial.csv')
testSet = pd.read_csv('./finalTestSetFirstTrial.csv')
trainSetU1=pd.read_csv('./uniqueU1_trainSet.csv')
testSetU1=pd.read_csv('./uniqueU1_testSet.csv')
trainSetU2=pd.read_csv('./uniqueU2_trainSet.csv')
testSetU2=pd.read_csv('./uniqueU2_testSet.csv')
trainSetU3=pd.read_csv('./uniqueU3_trainSet.csv')
testSetU3=pd.read_csv('./uniqueU3_testSet.csv')
trainSetU4=pd.read_csv('./uniqueU4_trainSet.csv')
testSetU4=pd.read_csv('./uniqueU4_testSet.csv')


#model
def predict(trainingSet,testSet,model):

    trainingSet=np.array(trainingSet)
    testSet=np.array(testSet)

    columns=['song_id','artist_id', 'Ds', 'published_time', 'song_init_plays','language','gender', 'play', 'download','collect','isWeekEnd','isMonday','isTuesday','isWednesday','isThursday','isFriday','dayAfterPublished','hasPublishedRecently']
    testSet=pd.DataFrame(columns=columns,data=testSet)
    ########here might be the problem:not testSet result didn't acorrds to artists but to songs
    testResult=testSet.groupby(['artist_id','Ds'],as_index=False).play.sum()
    testResult.to_csv('./testResultbyArtists.csv',index=False)
    testResult=np.array(testResult)
    
    testSet = np.array(testSet)
    
    X=trainingSet[:,[4,5,8,9,10,11,12,13,14,15,16,17]]
    Y=trainingSet[:,7]
    testData=testSet[:,[4,5,8,9,10,11,12,13,14,15,16,17]]
    #data preprocessing
    #scaler = preprocessing.MinMaxScaler().fit(X)  
    #X = scaler.transform(X)
    #testData = scaler.transform(testData)
    predictResult=0
    if model==1:
        gdbt=GradientBoostingRegressor()
        gdbt.fit(X,Y)
        rf=RandomForestRegressor()
        rf.fit(X,Y)
        predictResult1=rf.predict(testData)
        predictResult2=gdbt.predict(testData)
        predictResult=0.3*predictResult1+0.7*predictResult2
    if model==2:
        lass=Lasso()
        lass.fit(X,Y)
        predictResult=lass.predict(testData)
    finalResult=np.insert(testSet[:,[0,1,2]],3,values=predictResult,axis=1)
    columns=['songID','artistID','date','playcount']
    finalResult=pd.DataFrame(columns=columns,data=finalResult)
    finalResult=finalResult.groupby(['artistID','date'],as_index=False).playcount.sum()
    finalResult.to_csv('./predictResult.csv',index=False,header=False)
    finalResult=np.array(finalResult)
    #calculate score
    #userID used to initialize
    lastArtistID="023406156015ef87f99521f3b343f71f"
    uniqueUID=['2b7fedeea967becd9408b896de8ff903','c026b84e8f23a7741d9b670e3d8973f0','97de6333157f35467dff271d7afb0a23','8fb3cef29f2c266af4c9ecef3b780e97']
    scoreArtist=[]
    ratioArtist=[]
    artistScore=0
    artistRatio=0
    countN=0
    for i in range(0,len(finalResult)):

        if (finalResult[i][0]==lastArtistID or finalResult[i][0] in uniqueUID) and i!=(len(finalResult)-1):
            realValue=float(testResult[i][2])
            if realValue!=0:
                artistScore=artistScore+((float(finalResult[i][2])-realValue)/realValue)**2
            else:
                artistScore = artistScore + ((float(finalResult[i][2]) - realValue) / 1) ** 2
            artistRatio=artistRatio+testResult[i][2]
            countN=countN+1
            lastArtistID=finalResult[i][0]
        else:
            #print countN
            #print finalResult[i]
            
            artistScore=math.sqrt(float(artistScore)/countN)
            artistRatio=math.sqrt(artistRatio)
            scoreArtist.append(artistScore)
            ratioArtist.append(artistRatio)
            artistScore=0
            artistRatio=0
            countN=0
            realValue=float(testResult[i][2])
            if realValue!=0:
                artistScore=artistScore+((float(finalResult[i][2])-realValue)/realValue)**2
            else:
                artistScore = artistScore + ((float(finalResult[i][2]) - realValue) / 1) ** 2
            artistRatio = artistRatio + testResult[i][2]
            countN = countN + 1
            lastArtistID = finalResult[i][0]

    finalScore=0
    for i in range(0,len(scoreArtist)):
        finalScore=finalScore+(1-scoreArtist[i])*ratioArtist[i]
    print finalScore
    print scoreArtist
    print ratioArtist

if __name__ == '__main__':
    predict(trainingSet,testSet,1)
    predict(trainSetU1,testSetU1,1)
    predict(trainSetU2,testSetU2,1)
    predict(trainSetU3,testSetU3,1)
    predict(trainSetU4,testSetU4,1)