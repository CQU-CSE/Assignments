#utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ====================Part 1: Data Preprocessing====================
#  
#read the data
songs=pd.read_csv('p2_mars_tianchi_songs.csv',names=['song_id','artist_id','publish_time','song_init_plays','Language','Gender'])
users_actions=pd.read_csv('mars_tianchi_user_actions.csv',names=['user_id','song_id',' gmt_create','action_type','Ds'])

#separate the data from 7-8 month as testing set and others as training set
train_users_actions=users_actions[users_actions['Ds']<20150701]
train_songs=songs[songs['publish_time']<20150701]
test_users_actions=users_actions[users_actions['Ds']>=20150701]
test_songs=songs[songs['publish_time']>=20150701]

#merge the songs table to users_actions table on 'song_id'
test_set=pd.merge(test_users_actions,songs,how='left',on='song_id')
train_set=pd.merge(train_users_actions,songs,how='left',on='song_id')

#create plays of days of artist matrix for testing
artists=test_set['artist_id'].unique()
test_day_len=len(test_set['Ds'].unique())
test_artist_day_play_matrix=np.zeros([test_day_len,len(artists)])
predict_artist_day_play_matrix=np.zeros([test_day_len,len(artists)])
date=20150701
#for each day to be predicted
for day_index in range(test_day_len):
	if date==20150732:
		date=20150801
	set_of_date=test_set[test_set['Ds']==date]
    	#for each artist in current day
	for artist_index in range(len(artists)):
		artist_id=artists[artist_index]
        	set_of_date_of_artist=set_of_date[set_of_date['artist_id']==artist_id]
		#plays
		t = set_of_date_of_artist
		artist_plays = (t[t['action_type'] == 1])[['song_id', 'action_type']].groupby('song_id').sum().sum()[0]
		test_artist_day_play_matrix[day_index][artist_index]=artist_plays
	date+=1

#create plays of days of artist matrix for training
#artists in the training set are same to those in the testing set
train_day_len=len(train_set['Ds'].unique())
#from March to June 31+30+31+30=122 days
train_artist_day_play_matrix=np.zeros([train_day_len,len(artists)])
date=20150301
for day_index in range(train_day_len):
	if date==20150332:
		date=20150401
	elif date==20150431:
		date=20150501
	elif date==20150532:
		date=20150601
    	set_of_date = train_set[train_set['Ds'] == date]
	for artist_index in range(len(artists)):
		artist_id=artists[artist_index]
    		set_of_date_of_artist=set_of_date[set_of_date['artist_id']==artist_id]
    		# plays
		t=set_of_date_of_artist
		artist_plays=(t[t['action_type']==1])[['song_id','action_type']].groupby('song_id').sum().sum()[0]
        	train_artist_day_play_matrix[day_index][artist_index]=artist_plays
    	date += 1

#create mean plays of months of artist matrix for training
#from March to June
train_month_len=4
train_artist_month_play_matrix=np.zeros([train_month_len,len(artists)])
t=train_artist_day_play_matrix
#March
train_artist_month_play_matrix[0]=train_artist_day_play_matrix[0:31].mean(0)
#April
train_artist_month_play_matrix[1]=train_artist_day_play_matrix[31:61].mean(0)
#May
train_artist_month_play_matrix[2]=train_artist_day_play_matrix[61:92].mean(0)
#June
train_artist_month_play_matrix[3]=train_artist_day_play_matrix[92:122].mean(0)
#plot
for artist in range(len(artists)):
	x=[i+1 for i in range(train_month_len)]
	y=train_artist_month_play_matrix[:,artist]
	plt.plot(x,y)
	plt.show()
	keys=raw_input('press Enter to continue, ESC then Enter to stop: ')
	if keys=='\x1b':
		break

#create last four weeks mean plays of artist matrix for training
train_week_len=4
train_artist_last_week_play=np.zeros([train_month_len,len(artists)])
for i in range(train_week_len):
	train_artist_last_week_play[-i]=train_artist_day_play_matrix[train_day_len-(i+1)*7:train_day_len-i*7].mean(0)
#plot
for artist in range(len(artists)):
	x=[i+1 for i in range(train_week_len)]
	y=train_artist_last_week_play[:,artist]
	plt.plot(x,y)
	plt.show()
	keys=raw_input('press Enter to continue, ESC then Enter to stop: ')
	if keys=='\x1b':
		break


## ====================Part 3.0: Simply Predict with Last Days Mean Value====================
#  5437.835572344743
maxF=0
best_day=None
for day in range(1,train_day_len):
	last_days_mean_plays=train_artist_day_play_matrix[-day:].mean(0)
	#exception handle, testing_set_day_artist_matrix[24][-3] and [30][-3] are 0
	test_artist_day_play_matrix[24][-3]=1
	test_artist_day_play_matrix[30][-3]=1
	error=(test_artist_day_play_matrix - last_days_mean_plays) / test_artist_day_play_matrix
	#remove the exception point
	error[24][-3]=0
	error[30][-3]=0
	sigma=((error**2).sum(0)/test_day_len)**(0.5)
	phi=test_artist_day_play_matrix.sum(0)**(0.5)
	F=((1-sigma)*phi).sum()
	if maxF<F:
		maxF=F
		best_day=day
print 'Simply Predict with Last',best_day,'Days Mean Value,and F is ',str(maxF)
#update predict_artist_day_play_matrix
for i in range(test_day_len):
	predict_artist_day_play_matrix[i]=train_artist_day_play_matrix[-best_day:].mean(0)



## ====================Part 4.0: Classification====================
#
#月连续下降的艺人6	#5612.696205736612
first_two_month=train_artist_month_play_matrix[0:2].mean(0)
last_two_month=train_artist_month_play_matrix[2:].mean(0)
last_month=train_artist_month_play_matrix[-1]
last_but_one_month=train_artist_month_play_matrix[-2]
#整体差距距离
lamba1=40
#松弛量
lamba2=16
#前两个月高于后两个月
condition1=first_two_month-last_two_month>lamba1
#最后一个月在一定松弛内低于倒数第二个月，没有出现反弹的趋势
condition2=(last_month-lamba2)<last_but_one_month
condition=condition2 * condition1
descent_artist=np.zeros([train_month_len,len(condition[condition==True])])
for i in range(train_month_len):
	descent_artist[i]=train_artist_month_play_matrix[i][condition]
#plot
for artist in range(len(descent_artist[0])):
	x=[i+1 for i in range(train_month_len)]
	y=descent_artist[:,artist]
	plt.plot(x,y)
	plt.show()
	keys=raw_input('press Enter to continue, ESC then Enter to stop: ')
	if keys=='\x1b':
		break
#update predict_artist_day_play_matrix
alpha=0.8
for i in range(test_day_len):
	if i % 23 ==0:
		alpha*=0.95
	if i> 59:
		alpha=0.8
	predict_artist_day_play_matrix[i][condition]*=alpha

#月连续上升的艺人	#5670.70484768
first_two_month=train_artist_month_play_matrix[0:2].mean(0)
last_two_month=train_artist_month_play_matrix[2:].mean(0)
last_month=train_artist_month_play_matrix[-1]
last_but_one_month=train_artist_month_play_matrix[-2]
#整体差距距离
lamba1=40
#松弛量
lamba2=10
#前两个月低于后两个月
condition1=last_two_month-first_two_month>lamba1
#最后一个月在一定松弛内高于倒数第二个月，没有出现下滑的趋势
condition2=(last_month+lamba2)>last_but_one_month
condition=condition2 * condition1
ascent_artist=np.zeros([train_month_len,len(condition[condition==True])])
for i in range(train_month_len):
	ascent_artist[i]=train_artist_month_play_matrix[i][condition]
#plot
for artist in range(len(ascent_artist[0])):
	x=[i+1 for i in range(train_month_len)]
	y=ascent_artist[:,artist]
	plt.plot(x,y)
	plt.show()
	keys=raw_input('press Enter to continue, ESC then Enter to stop: ')
	if keys=='\x1b':
		break
#update predict_artist_day_play_matrix
alpha=1.0
for i in range(test_day_len):
	if i % 2 ==0:
		alpha*=1.014
	if i> 15:
		alpha=0.974
	predict_artist_day_play_matrix[i][condition]*=alpha

#星期趋势，每周末的播放量比较低	#5660.63702108
sigma=0.91
first_weekend_index=4
while first_weekend_index<test_day_len:
	predict_artist_day_play_matrix[first_weekend_index,:]*=sigma
	first_weekend_index+=7
first_weekend_index=3
while first_weekend_index<test_day_len:
	predict_artist_day_play_matrix[first_weekend_index,:]*=sigma
	first_weekend_index+=7


## ====================Part END: Final Predict F====================
#
error=(test_artist_day_play_matrix - predict_artist_day_play_matrix) / test_artist_day_play_matrix
#remove the exception point
error[24][-3]=0
error[30][-3]=0
sigma=((error**2).sum(0)/test_day_len)**(0.5)
phi=test_artist_day_play_matrix.sum(0)**(0.5)
F=((1-sigma)*phi).sum()
print 'Final F is ',str(F)
