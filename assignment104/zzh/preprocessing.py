#import dataset
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from numpy import *



songs = csv.reader(open('/Users/zhaozehua/Desktop/ali music/p2_mars_tianchi_songs.csv','r'))
trainingset = csv.reader(open('/Users/zhaozehua/Desktop/ali music/training.csv','r'))
testingset = csv.reader(open('/Users/zhaozehua/Desktop/ali music/testing.csv','r'))

date_l=[datetime.strftime(x,'%Y%m%d') for x in list(pd.date_range(start='20150301', end='20150831'))] # for index
songlist = {} # song information,including artist, public time, init_plays, language
artistlist = [] #for index
artist = {} #artist information, init_plays and gender
newsong = np.zeros([100,184]) #new song information(init_plays), each row is a artist, each col represents the data in data_l
list = [] #error list

for line in songs :
    if line[2] < '20150901':
        songlist[line[0]] = line[1:4]
        if line[1] in artistlist:
            if line[2] in date_l:
                col = date_l.index(line[2])
                row = artistlist.index(line[1])
                newsong[row][col] += int(line[3])
            if line[2] < '20150301':
                artist[line[1]][0] = artist[line[1]][0] + int(line[3])
        else:
            artistlist.append(line[1])
            if line[2] in date_l:
                col = date_l.index(line[2])
                row = artistlist.index(line[1])
                newsong[row][col] += int(line[3])
                if line[5] == '1':
                    artist[line[1]] = [0, [1,0,0]]
                if line[5] == '2':
                    artist[line[1]] = [0, [0,1,0]]
                if line[5] == '3':
                    artist[line[1]] = [0, [0,0,1]]
            if line[2] < '20150301' :
                if line[5] == '1':
                    artist[line[1]] = [int(line[3]), [1,0,0]]
                if line[5] == '2':
                    artist[line[1]] = [int(line[3]), [0,1,0]]
                if line[5] == '3':
                    artist[line[1]] = [int(line[3]), [0,0,1]]
    else:
        list.append(line[0])
list.append('9e57bfc704d837b492')


play_train = np.zeros([100,122])
collect_train = np.zeros([100,122])
download_train = np.zeros([100,122])

play_test = np.zeros([100,62])
collect_test = np.zeros([100,62])
download_test = np.zeros([100,62])

for line in trainingset:
    if line[1] not in list:
        artistid = songlist[line[1]][0]
        row = artistlist.index(artistid)
        col = date_l.index(line[4])
        if line[3] == '1':
            play_train[row][col] += 1
        if line[3] == '2':
            download_train[row][col] += 1
        if line[3] == '3':
            collect_train[row][col] += 1

for line in testingset:
    if line[1] not in list:
        artistid = songlist[line[1]][0]
        row = artistlist.index(artistid)
        col = date_l.index(line[4]) - 122
        if line[3] == '1':
            play_test[row][col] += 1
        if line[3] == '2':
            download_test[row][col] += 1
        if line[3] == '3':
            collect_test[row][col] += 1


count = -1
while(count < artistlist.__len__()-1):
    count += 1
    if any(play_train[count]) == False  and any(collect_train[count]) == False and any(download_train[count]) == False:
        artistlist.remove(artistlist[count])
        newsong = np.delete(newsong, count ,0)
        play_train = np.delete(play_train, count, 0)
        collect_train = np.delete(collect_train, count ,0)
        download_train = np.delete(download_train, count, 0)
        play_test = np.delete(play_test, count , 0)
        collect_test = np.delete(collect_test, count , 0)
        download_test = np.delete(download_test , count , 0)
        count -= 1

'''
num = 0
x = np.arange(0, 122)
while(num < 9):
    plt.subplot(3,3,1 + num)
    y = play_train[num]
    y1 = collect_train[num]
    y2 = download_train[num]
    y3 = newsong[num,0:122]

    plt.plot(x, y, markerfacecolor='blue')
    plt.plot(x, y1, markerfacecolor='red')
    plt.plot(x, y2, markerfacecolor='black')
    plt.plot(x, y3, markerfacecolor='yellow')
    plt.legend()
    num += 1
plt.savefig('/Users/zhaozehua/Desktop/ali music/1.png')
'''
