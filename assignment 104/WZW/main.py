import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#记录了每个歌手每天的播放量，下载量和收藏量
artistsplaybackSet = []
songsplaybackSet = []
usertrainSet = []
songsSet = []
list_artists_timevalue_playbacks = {}
list_songs_timevalue_playbacks = {}


def model(songlist):
    X = []
    Y = []
    X_test = []
    Y_test = []
    for i in range(len(songlist)):
        if i <= 120:
            x = []
            x.append(songlist[i][0])
            #x.append(float(songlist[i][1]))
            x.append(float(songlist[i][2]))
            x.append(float(songlist[i][3]))
            '''
            x.append(float(songlist[-2]))
            x.append(float(songlist[-3]))
            '''
            x.append(float(songlist[-1]) * songlist[i][0])
            
            X.append(x)
            y = []
            y.append(float(songlist[i][1]))
            Y.append(y)
        elif i > 120 and i <= 182:
            x = []
            x.append(songlist[i][0])
            #x.append(float(songlist[i][1]))
            x.append(float(songlist[i][2]))
            x.append(float(songlist[i][3]))
            '''
            x.append(float(songlist[-2]))
            x.append(float(songlist[-3]))
            '''
            x.append(float(songlist[-1]) * songlist[i][0])
            
            X_test.append(x)
            y = []
            y.append(float(songlist[i][1]))
            Y_test.append(y)
    
    quadratic_featurizer = PolynomialFeatures(degree=1)
    X_train_quadratic = quadratic_featurizer.fit_transform(X)
    X_test_quadratic = quadratic_featurizer.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_quadratic, Y)
    predictions = model.predict(X_test_quadratic)
    
    quadratic_featurizer1 = PolynomialFeatures(degree=2)
    X_train_quadratic1 = quadratic_featurizer1.fit_transform(X)
    X_test_quadratic1 = quadratic_featurizer1.transform(X_test)
    model1 = LinearRegression()
    model1.fit(X_train_quadratic1, Y)
    predictions1 = model1.predict(X_test_quadratic1)
    
    
    return predictions,predictions1

def strtofloat(str):
    l = []
    for i in str[0]:
        l.append(float(i))
    return l
    
#读取数据，csv
def reader(filename):
    l = []
    csv_reader = csv.reader(open(filename, 'r'))
    for row in csv_reader:
        l.append(row)
    return l

#读取播放量
def readplayback():
    with open('./artist_p_d_c.txt') as f:
        for line in f:
            line = line.strip()
            trainlist = line.split(',')
            artistsplaybackSet.append(trainlist)
            
    with open('./song_p_d_c.txt') as f:
        for line in f:
            line = line.strip()
            trainlist = line.split(',')
            songsplaybackSet.append(trainlist)
        

#歌手：{[日期，播放量]}
def inttimevalue(playbackSet):
    list_timevalue_playbacks = {}
    for i in range(len(playbackSet)):
        if (i % 4) == 0:
            list_timevalue_playbacks[playbackSet[i][0]] = []
        if (i % 4) == 1:
            #歌曲总播放量超过200和总收藏数超过10的才计算
            sum = 0
            sum1 = 0
            for time in range(len(playbackSet[i])):
                sum += int(playbackSet[i][time])
                sum1 += int(playbackSet[i+1][time])
            if sum > 200 and sum1 > 10:
                for j in range(len(playbackSet[i])):
                    temp_j = 0
                    temp = [20150301,20150400,20150500,20150600,20150700,20150800]
                    a = 0
                    list_temp = []
                    if (j - 30) <= 0:
                        a = 0
                        temp_j = j
                    elif (j - 30) > 0 and (j -60) <= 0:
                        a = 1
                        temp_j = j - 30
                    elif (j - 60) > 0 and (j -91) <= 0:
                        a = 2
                        temp_j = j - 60
                    elif (j - 91) > 0 and (j - 121) <= 0:
                        a = 3
                        temp_j = j - 91
                    elif (j - 121) > 0 and (j - 152) <= 0:
                        a = 4
                        temp_j = j - 121
                    elif (j - 152) > 0 and (j -183) <= 0:
                        a = 5
                        temp_j = j - 152
                    list_temp.append(float(temp[a])+float(temp_j))
                    list_temp.append(playbackSet[i][j])
                    list_temp.append(playbackSet[i+1][j])
                    list_temp.append(playbackSet[i+2][j])
                    list_timevalue_playbacks[playbackSet[i-1][0]].append(list_temp)
            else:
                del list_timevalue_playbacks[playbackSet[i-1][0]]

    return list_timevalue_playbacks

def initpredictartistsplay(playbackSet):
    list_timevalue_playbacks = {}
    for i in range(len(playbackSet)):
        if (i % 4) == 0:
            list_timevalue_playbacks[playbackSet[i][0]] = []
        if (i % 4) == 1:
            l = playbackSet[i][121:]
            list_timevalue_playbacks[playbackSet[i-1][0]].append(l)
    return list_timevalue_playbacks
    
#构建特征的LIST,歌曲：[时间差（播放时间-发布时间），初始热度]
def init(songset,songsplayback):
    dataset = songsplayback
    for key in songsplayback:
        for i in range(len(songsSet)):
            if key == songsSet[i][0]:
                dataset[key].append(songsSet[i][4])
                dataset[key].append(songsSet[i][5])
                dataset[key].append(songsSet[i][2])
                dataset[key].append(songsSet[i][3])
        
                    

    for key in dataset:
        for j in range(len(dataset[key])-4):
            dataset[key][j][0] =dataset[key][j][0] - float(dataset[key][-2])
        #print key,dataset[key][-2]
        del dataset[key][-2]

    return dataset



readplayback()
#list_artists_timevalue_playbacks = inttimevalue(artistsplaybackSet)

list_songs_timevalue_playbacks = inttimevalue(songsplaybackSet)

'''
实际的7,8月艺人的播放量，list_artists_predict
'''
list_artists_78 = initpredictartistsplay(artistsplaybackSet)

filename_songs = 'p2_mars_tianchi_songs.csv'
songsSet = reader(filename_songs)
#print list_songs_timevalue_playbacks[songsSet[0][0]][0][0]
dataSet = init(songsSet,list_songs_timevalue_playbacks)


predictsong = {}
predictsong1 = {}
for key in dataSet:
    predictions,predictions1 = model(dataSet[key])
    predictsong[key] = []
    predictsong[key].append(predictions)
    predictsong1[key] = []
    predictsong1[key].append(predictions1)
    

predictartists = {}
for key in predictsong:
    for temp in songsSet:
        if temp[0] == key:
            if predictartists.has_key(temp[1]):
                predictartists[temp[1]] = np.array(predictartists[temp[1]]) + np.array(predictsong[key])
            else:
                predictartists[temp[1]] = predictsong[key]

predictartists1 = {}
for key in predictsong1:
    for temp in songsSet:
        if temp[0] == key:
            if predictartists1.has_key(temp[1]):
                predictartists1[temp[1]] = np.array(predictartists1[temp[1]]) + np.array(predictsong1[key])
            else:
                predictartists1[temp[1]] = predictsong1[key]          

F = 0
aaa = 0
for artist in predictartists:
    predict_playbacks = predictartists[artist]
    predict_playbacks1 = predictartists1[artist]
    true_playbacks = strtofloat(list_artists_78[artist])
    
    one = ((np.array(predict_playbacks[0]).T - np.array(true_playbacks))/np.array(true_playbacks))[0]
    two = one ** 2
    theta = (sum(two)/len(true_playbacks))** 0.5
    if theta >= 1:
        all = 0
        for i in range(len(true_playbacks)):
            if true_playbacks[i] == 0:
                continue
            else:
                first = (predict_playbacks[0][i][0] - true_playbacks[i])/true_playbacks[i]
                second = first ** 2
        all += second
        theta = (all / len(true_playbacks)) ** 0.5
        print artist
        
    one1 = ((np.array(predict_playbacks1[0]).T - np.array(true_playbacks))/np.array(true_playbacks))[0]
    two1 = one1 ** 2
    theta1 = (sum(two1)/len(true_playbacks))** 0.5
    if theta1 >= 1:
        all = 0
        for i in range(len(true_playbacks)):
            if true_playbacks[i] == 0:
                continue
            else:
                first = (predict_playbacks1[0][i][0] - true_playbacks[i])/true_playbacks[i]
                second = first ** 2
        all += second
        theta1 = (all / len(true_playbacks)) ** 0.5
        print artist
    print theta,theta1
    if theta > theta1:
        theta = theta1
    print theta
    flag = (sum(np.array(true_playbacks)))**0.5
    
    F += (1-theta)*flag
    print F







    


