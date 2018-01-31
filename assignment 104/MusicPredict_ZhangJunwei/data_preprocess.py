#-*- coding:utf8 -*-#
import os
import csv
import time
from collections import defaultdict

####################### date ####################################################
# map date into num
# date
print ""
print "===start generate date rank====="
date2rank = {}
rank2date = {}
import datetime
dt = datetime.datetime(2015, 03, 01, 00, 00, 01)
end = datetime.datetime(2015, 10, 30, 23, 59, 59)
step = datetime.timedelta(days=1)
day_rank = 0
while dt < end:
    day_date = dt.strftime('%Y%m%d')
    rank2date[day_rank] = day_date
    date2rank[day_date] = day_rank
    dt += step
    day_rank += 1
print "date num ", len(rank2date)
#print "rank to date :", rank2date
print "===end generate date rank====="

####################### songs ####################################################
# load songs date
song_id_set = set()
songs_id2songinfo = defaultdict(tuple)
songs_rank2iddate = [] #song rank to song_id and publish_date
songs_id2rank = {}

artist_id_set = set()
artists_rank2id = []
artists_id2rank = {}

print "===start load songs====="
t0 = time.time()

song_file_path = "songs.csv"
f = open(song_file_path, 'r')
rows = csv.reader(f)

for row in rows:
    song_id = row[0]
    song_id_set.add(song_id)

    artist_id = row[1]
    artist_id_set.add(artist_id)

    publish_time = int(row[2])
    songs_rank2iddate.append((song_id, publish_time))
    songs_id2songinfo[song_id] = (artist_id, publish_time)

# rank songs by date
songs_rank2iddate.sort(key = lambda item : item[1])
for rank, item in enumerate(songs_rank2iddate):
    songs_id2rank[item[0]] = rank

artists_rank2id = list(artist_id_set)
for rank, item in enumerate(artists_rank2id):
    artists_id2rank[item] = rank
artists_rank2id = list(artist_id_set)

print "songs num ", len(song_id_set)
print "songs_id2songinfo num ", len(songs_id2songinfo)
print "artist num ", len(artist_id_set)
# print "k th artist songs num ", artists_rank2songs_num

t1 = time.time()
print "It takes %f s to load songs" %(t1-t0)
print "===end load songs===="

####################### actions ####################################################
# load songs actions
user_id_set = set()
song_hasact_id_set = set()
action_type_set = set()

print "===start user statistics==="
tu0 = time.time()
ua_file_path1 = "users.csv"
f1 = open(ua_file_path1, 'r')
rows1 = csv.reader(f1)
for idx, row in enumerate(rows1):
    user_id = row[0]
    user_id_set.add(user_id)

    song_id = row[1]
    song_hasact_id_set.add(song_id)

    action_type = int(row[3])
    action_type_set.add(action_type)
tu1 = time.time()
print "user num", len(user_id_set)
print "song num that has action", len(song_hasact_id_set)
print "action type num", len(action_type_set)
print "It takes %f s to do user statistics" %(tu1-tu0)
print "===end user statistics==="

####################### actions statistics####################################################
artists_play = defaultdict(list)
artists_play_inday = defaultdict(list)
print ""
print "===start action statistics====="
ta0 = time.time()

ua_file_path = "users.csv"
f = open(ua_file_path, 'r')
rows = csv.reader(f)
for idx, row in enumerate(rows):
    user_id = row[0]
    #user_rank = users_id2rank[user_id]

    song_id = row[1]
    song_rank = songs_id2rank[song_id]
    artist_rank = artists_id2rank[songs_id2songinfo[song_id][0]]

    action_time_hour = int(row[2])

    action_type = int(row[3])

    action_time_date = date2rank[row[4]]

    if(action_type == 1):
        artists_play[artist_rank].append((action_time_hour, action_time_date))

# k是artists的编号， v为歌曲播放时间及记录时间
for k, v in artists_play.items():
    v.sort(key = lambda item : item[1])
    artists_play[k] = v

for k, v in artists_play.items():
    vd = []
    c = 1
    dateTemp = -1
    itemTemp = (0, 0)
    for item in v:
        if(item[1] == dateTemp):
            c += 1
        else:
            vd.append((c, itemTemp[1]))
            dateTemp = item[1]
            itemTemp = item
            c = 1
    vd.append((c, itemTemp[1]))
    vd.pop(0)
    artists_play_inday[k] = vd
artists_play.clear()

ta1 = time.time()
print "the artists number:", len(artists_play_inday)
print "It takes %f s to do action statistics" %(ta1-ta0)
print "===end actions statistics====="
