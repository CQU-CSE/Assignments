# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def dataCleaning():
    dataSet=pd.read_csv('./trainset.csv')
    dataSet=np.array(dataSet)
    #length before cleaning 
    print len(dataSet)
    lastUserID='default'
    lastSongID='default'
    lastUserAction='1'
    #record user action two times ago,if a user download or collect songs in a short time,consider it unusual 
    twoTimesUserAction='1'
    time=0
    for idx, val in enumerate(dataSet):
        index=idx
        userID=val[1]
        songID=val[2]
        userAction=val[3]
        lluserAction=lastUserAction
         #delete the situation that only listen to one song in a short time
         #delete the situation that users are continuously downloading and collecting in a short time
        if ((val[1]==lastUserID and val[2]==lastSongID) or (val[3]!=1 and val[3]==lastUserAction and val[3]==twoTimesUserAction)) :
            dataSet= np.delete(dataSet, index, axis=0)
        lastUserID=userID
        lastSongID=songID
        lastUserAction=userAction
        twoTimesUserAction=lluserAction
        time+=1
        print 'progress:%d/%d'%(time,len(dataSet))
    print len(dataSet)
    columns=['user_id','song_id','gmt_create','action_type','Ds']
    pdTrainSet=pd.DataFrame(columns=columns,data=dataSet)
    pdTrainSet.to_csv('./trainSetAfterCleaning.csv',index=False,header=False)
    
dataCleaning()