import pandas as pd
import numpy as np
import datetime

#feature from artist-song perspective

#one hot encoding##############Important
def one_hot(pdframe,rowName):
    rows=pd.get_dummies(pdframe[rowName])
    pdframe=pdframe.drop(rowName,axis=1)
    pdframe=pdframe.join(rows)
    return pdframe

#calculate the recent date of publishedsongs
def addDate(recentDate,deltaday):
    startDate=str(recentDate)
    startDate=datetime.datetime.strptime(startDate,'%Y%m%d')
    date = startDate + datetime.timedelta(days = deltaday)
    date=date.strftime('%Y%m%d')
    return long(date)

#preprocessing artist-song relationships
def SplittestSet():

    dataSet=pd.read_csv('./testset.csv')
    dataSet=np.array(dataSet)
    columns = ['user_id', 'song_id', 'gmt_create', 'action_type', 'Ds']
    pdUser = pd.DataFrame(columns=columns, data=dataSet)
    #do one hot encoding to the row"action_type"
    pdUser = one_hot(pdUser,"action_type")
    #delete the gmt_create and the user row which is of no use if we anylize our problem in artist-song's perspective
    pdUser = pdUser.drop('gmt_create', axis = 1)
    pdUser = pdUser.drop('user_id', axis = 1)

    #Data aggregation(on User-song record(everyday))##############important#####################
    pdUserGBySong=pdUser.groupby(['song_id','Ds',1,2,3]).size().reset_index(name='count')
    pdUserGBySong = np.array(pdUserGBySong)
    columns = ['song_id', 'Ds', 'play', 'download','collect','actionCount']
    finalUserGBySong = pd.DataFrame(columns=columns, data=pdUserGBySong)
    #merge the user record and the song record
    artist_song=pd.read_csv('./p2_mars_tianchi_songs.csv')
    artist_song=np.array(artist_song)
    columns = ['song_id', 'artist_id', 'published_time', 'song_init_plays', 'language', 'gender']
    artist_song=pd.DataFrame(columns=columns, data=artist_song)
    finalTrainSetWithoutDataCleaning = pd.merge(artist_song, finalUserGBySong, on='song_id')
    finalTrainSetWithoutDataCleaning = finalTrainSetWithoutDataCleaning.groupby(['song_id' ,'artist_id','Ds','play','download','collect','published_time', 'song_init_plays','language','gender'],as_index=False).actionCount.sum()
    #adding all the actions together
    finalTS=np.array(finalTrainSetWithoutDataCleaning)
    finalTS[:, 3] = finalTS[:, 3] * finalTS[:, 10]
    finalTS[:, 4] = finalTS[:, 4] * finalTS[:, 10]
    finalTS[:, 5] = finalTS[:, 5] * finalTS[:, 10]
    finalTS=np.delete(finalTS,10,axis=1)
    #combine play,download and collect
    columns1 = ['song_id','artist_id','Ds','play','download','collect','published_time', 'song_init_plays','language','gender']
    finalTS=pd.DataFrame(columns=columns1,data=finalTS)
    download = finalTS.groupby(['song_id','artist_id', 'Ds', 'published_time', 'song_init_plays','language','gender'], as_index=False).download.sum()[['download']]
    collect = finalTS.groupby(['song_id','artist_id', 'Ds','published_time', 'song_init_plays','language', 'gender'], as_index=False).collect.sum()[['collect']]
    finalTS = finalTS.groupby(['song_id','artist_id', 'Ds', 'published_time', 'song_init_plays','language','gender'], as_index=False).play.sum()
    finalTS = finalTS.join(download)
    finalTS=finalTS.join(collect)
    #new feature is weekend,is weekday,day til published
    finalTS=np.array(finalTS)
    isWeekend=np.zeros(len(finalTS))
    isMonday=np.zeros(len(finalTS))
    isTuesday=np.zeros(len(finalTS))
    isWednesday=np.zeros(len(finalTS))
    isThursday=np.zeros(len(finalTS))
    isFriday=np.zeros(len(finalTS))
    dayAfterPublished=np.zeros(len(finalTS))
    artistRecentPublished=[]
    for i in range(0,len(finalTS)):
        pbday=str(finalTS[i][3])
        today=str(finalTS[i][2])
        pbday=datetime.datetime.strptime(pbday,'%Y%m%d')
        today=datetime.datetime.strptime(today,'%Y%m%d')
        deltaday=today-pbday
        if(deltaday.days>=0):
            dayAfterPublished[i]=deltaday.days#feature dayAfterPublished
        else:
            dayAfterPublished[i]=-1
        if (1<=finalTS[i][2]-finalTS[i][3]<=10):
            artistName=finalTS[i][1]
            Recentdate=finalTS[i][2]
            for t in range(0,10):
                receivedDate=addDate(Recentdate,t)
                artistRecentPublished.append([artistName,receivedDate])
            '''
            artistRecentPublished.append([finalTS[i][1],finalTS[i][2]])#record artist name and date that was near after publish(record the following 10days)
            artistRecentPublished.append([finalTS[i+1][1],finalTS[i+1][2]])
            artistRecentPublished.append([finalTS[i+2][1],finalTS[i+2][2]])
            artistRecentPublished.append([finalTS[i+3][1],finalTS[i+3][2]])
            artistRecentPublished.append([finalTS[i+4][1],finalTS[i+4][2]])
            artistRecentPublished.append([finalTS[i+5][1],finalTS[i+5][2]])
            artistRecentPublished.append([finalTS[i+6][1],finalTS[i+6][2]])
            artistRecentPublished.append([finalTS[i+7][1],finalTS[i+7][2]])
            artistRecentPublished.append([finalTS[i+8][1],finalTS[i+8][2]])
            '''
            
        time1=str(finalTS[i][2])
        week = datetime.datetime.strptime(time1, "%Y%m%d").weekday()
        if(week==5 or week==6):
            isWeekend[i]=1
        if week==0:
            isMonday[i]=1
        if week==1:
            isTuesday[i]=1
        if week==2:
            isWednesday[i]=1
        if week==3:
            isThursday[i]=1
        if week==4:
            isFriday[i]=1
    finalTS=np.insert(finalTS,10, values=isWeekend, axis=1)
    finalTS=np.insert(finalTS,11, values=isMonday, axis=1)
    finalTS=np.insert(finalTS,12, values=isTuesday, axis=1)
    finalTS=np.insert(finalTS,13, values=isWednesday, axis=1)
    finalTS=np.insert(finalTS,14, values=isThursday, axis=1)
    finalTS=np.insert(finalTS,15, values=isFriday, axis=1)
    finalTS=np.insert(finalTS,16, values=dayAfterPublished, axis=1)
    
    #add to testSet
    hasPublishedRecently=np.zeros(len(finalTS))
    finalTS=np.insert(finalTS,17, values=hasPublishedRecently, axis=1)
    
    for i in range(0,len(finalTS)):
        for j in range(0,len(artistRecentPublished)):    
            if finalTS[i][1]==artistRecentPublished[j][0] and finalTS[i][2]==artistRecentPublished[j][1]:
                finalTS[i][17]=1  
    
    
    #get a unique user
    uniqueUser1=[]
    deleteList1=[]#record the index in finalTs we need to delete
    #uniqueUser bellow are important users
    uniqueUser2=[]
    deleteList2 = []
    uniqueUser3=[]
    deleteList3 = []
    uniqueUser4=[]
    deleteList4 = []  
    for i in range(0,len(finalTS)):
        if finalTS[i][1]=='2b7fedeea967becd9408b896de8ff903':
            uniqueUser=finalTS[i]
            deleteList1.append(i)
            uniqueUser1.append(uniqueUser)
    for num in range(0,len(deleteList1)):
        finalTS=np.delete(finalTS,deleteList1[num]-num,0)#previously I wrote finalTS=np.delete(finalTS,deleteList[num],0) and it was wrong,cause num is increasing while the listlen is decreasing at the same time
            
    for i in range(0,len(finalTS)):    
        if finalTS[i][1]=='c026b84e8f23a7741d9b670e3d8973f0':
            uniqueUser=finalTS[i]
            deleteList2.append(i)
            uniqueUser2.append(uniqueUser)
    for num in range(0,len(deleteList2)):
        finalTS=np.delete(finalTS,deleteList2[num]-num,0)
                   
    for i in range(0,len(finalTS)):
        if finalTS[i][1]=='97de6333157f35467dff271d7afb0a23':
            uniqueUser=finalTS[i]
            deleteList3.append(i)
            uniqueUser3.append(uniqueUser)
    for num in range(0,len(deleteList3)):
        finalTS=np.delete(finalTS,deleteList3[num]-num,0)
        
    for i in range(0, len(finalTS)):
        if finalTS[i][1]=='8fb3cef29f2c266af4c9ecef3b780e97':
            uniqueUser=finalTS[i]
            deleteList4.append(i)
            uniqueUser4.append(uniqueUser)
    for num in range(0,len(deleteList4)):
        finalTS=np.delete(finalTS,deleteList4[num]-num,0)
        
    #output unique users
    columnsU1=['song_id','artist_id', 'Ds', 'published_time', 'song_init_plays','language','gender', 'play', 'download','collect','isWeekEnd','isMonday','isTuesday','isWednesday','isThursday','isFriday','dayAfterPublished','hasPublishedRecently']
    uniqueUser1=pd.DataFrame(columns=columnsU1,data=uniqueUser1)
    uniqueUser1.to_csv('./uniqueU1_testSet.csv',header=False,index=False)
    
    columnsU2=['song_id','artist_id', 'Ds', 'published_time', 'song_init_plays','language','gender', 'play', 'download','collect','isWeekEnd','isMonday','isTuesday','isWednesday','isThursday','isFriday','dayAfterPublished','hasPublishedRecently']
    uniqueUser2=pd.DataFrame(columns=columnsU2,data=uniqueUser2)
    uniqueUser2.to_csv('./uniqueU2_testSet.csv',header=False,index=False)
    
    columnsU3=['song_id','artist_id', 'Ds', 'published_time', 'song_init_plays','language','gender', 'play', 'download','collect','isWeekEnd','isMonday','isTuesday','isWednesday','isThursday','isFriday','dayAfterPublished','hasPublishedRecently']
    uniqueUser3=pd.DataFrame(columns=columnsU3,data=uniqueUser3)
    uniqueUser3.to_csv('./uniqueU3_testSet.csv',header=False,index=False)
    
    columnsU4=['song_id','artist_id', 'Ds', 'published_time', 'song_init_plays','language','gender', 'play', 'download','collect','isWeekEnd','isMonday','isTuesday','isWednesday','isThursday','isFriday','dayAfterPublished','hasPublishedRecently']
    uniqueUser4=pd.DataFrame(columns=columnsU4,data=uniqueUser4)
    uniqueUser4.to_csv('./uniqueU4_testSet.csv',header=False,index=False)
    
    #output testSet
    columns3=['song_id','artist_id', 'Ds', 'published_time', 'song_init_plays','language','gender', 'play', 'download','collect','isWeekEnd','isMonday','isTuesday','isWednesday','isThursday','isFriday','dayAfterPublished','hasPublishedRecently']
    finalTS=pd.DataFrame(columns=columns3, data=finalTS)
    finalTS.to_csv('./finalTestSetFirstTrial.csv',header=False,     index=False)

if __name__ == '__main__':
    SplittestSet()