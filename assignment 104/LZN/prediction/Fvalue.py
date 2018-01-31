# -*-coding:utf-8-*-
import math

artist_predict={}
for line in open('../dataset/prediction_del_ds_add_weekandfestival_feature_final_int.csv','r'):
    (artistid,date,playnum)=line.split(',')
    artist_predict.setdefault(artistid,{})
    artist_predict[artistid][date]=float(playnum)
# print artist_predict

artist_true={}
for line in open('../dataset/truevalue.csv','r'):
    (artistid, date, playnum)=line.split(',')
    artist_true.setdefault(artistid,{})
    artist_true[artistid][date]=float(playnum)

artist_fangcha=[]
artist_quanzhong=[]

for artist in artist_predict:
    zonghe=0
    bofangliangSum=0
    for date in artist_predict[artist]:
        chazhi=((artist_predict[artist][date]-artist_true[artist][date])/(artist_true[artist][date]))
        pingfang=chazhi**2
        bofangliang=artist_true[artist][date]
        zonghe+=pingfang
        bofangliangSum+=bofangliang
    fangcha=math.sqrt(zonghe/len(artist_predict[artist]))
    quanzhong=math.sqrt(bofangliangSum)
    artist_fangcha.append(fangcha)
    artist_quanzhong.append(quanzhong)

sumf=0
for i in range(len(artist_quanzhong)):
    singlef=(1-artist_fangcha[i])*artist_quanzhong[i]
    sumf+=singlef
print sumf