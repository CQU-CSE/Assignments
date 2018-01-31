########################## fit ##################################################
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from data_preprocess import *

artists_num = len(artists_play_inday)
ap_dfs = {}
d = []
date_rank = range(0, 183)
for i in range(0, 183):
    d.append(np.datetime64(pd.to_datetime(rank2date[date_rank[i]], format='%Y%m%d').to_datetime()))
d2 = []
date_rank2 = range(0, 244)
for i in range(0, 244):
    d2.append(np.datetime64(pd.to_datetime(rank2date[date_rank2[i]], format='%Y%m%d').to_datetime()))
    
def APtoDF():
    ap_dfs.clear()
    for j in range(0, artists_num):
        p = artists_play_inday[j]
        c = [80] * 183 # 6个的时间
        for i in p:
            ci = i[0]  # 歌曲播放时间
            if ci <= 0:
                ci = 80
            c[i[1]] = ci  # 歌曲记录时间：歌曲播放时间
        # 形成一个一维矩阵
        s = pd.Series(np.array(c, dtype=np.float64), index=d, name='artist' + str(j) + 'play')
        ap_dfs[j] = s
    ap_df = pd.DataFrame(ap_dfs)
    return ap_df

def predstl():
    frac_l = {}
    print 'predict the all artists'
    frac_l.clear()
    predict_file_path = "predict.csv"
    fp = open(predict_file_path, 'wb')
    fpwriter = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    split_date_rank = 122
    F = []
    for j in range(0, artists_num):
        print 'artists:', j
        plt.close('all')
        plt.figure(j)
        #ap_df[j].plot()
        orig = np.log(ap_df[j])[:split_date_rank]  #对播放量进行log变换，减少振幅
        orig_arr = np.asarray((ap_df[j]).tolist())
        # 星期季节趋势
        stl_w = sm.tsa.seasonal_decompose(orig.tolist(), freq=7)  # 进行分解
        stl_w_se = stl_w.seasonal   # 季节趋势（除此之外，还有长期趋势和随机成分）
        w_s = stl_w_se[-7:]
        # 月份季节趋势
        stl_w_rest = orig - stl_w_se
        stl_m = sm.tsa.seasonal_decompose(np.nan_to_num(stl_w_rest).tolist(), freq=30)
        stl_m_se = stl_m.seasonal
        m_s = stl_m_se[-30:]
        # 对星期和月份季节趋势分解之后的时间序列
        rest = stl_w_rest - stl_m_se
        # 将数据转换为series
        rest_s = pd.Series(rest, index=d, name='artist' + str(j) + 'rest')
        rest_x = range(0, split_date_rank)
        #sp = [11, 19, 25, 30, 31, 33, 34, 43, 51, 62, 63]
        sp = [11]
        if j in sp:
            frac_l[j] = 0.05
            rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.05, return_sorted=False)
            rest_ss = pd.Series(rest_as, index=d[:split_date_rank], name='artist' + str(j) + 'rest')

            order = (6, 0, 1)
            model  = ARIMA(rest_ss, order, freq='D')
            model = model.fit()
            model.predict(1, 255)
        else:
            try:
                frac_l[j] = 0.2
                rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.2, return_sorted=False)
                rest_ss = pd.Series(rest_as, index=d[:split_date_rank], name='artist' + str(j) + 'rest')

                order = (8, 0, 1)
                model  = ARIMA(rest_ss, order, freq='D')
                model = model.fit()
                model.predict(1, 255)
            except:
                try:
                    frac_l[j] = 0.18
                    rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.18, return_sorted=False)
                    rest_ss = pd.Series(rest_as, index=d[:split_date_rank], name='artist' + str(j) + 'rest')

                    order = (8, 0, 1)
                    model  = ARIMA(rest_ss, order, freq='D')
                    model = model.fit()
                    model.predict(1, 255)
                except:
                    try:
                        frac_l[j] = 0.16
                        rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.16, return_sorted=False)
                        rest_ss = pd.Series(rest_as, index=d[:split_date_rank], name='artist' + str(j) + 'rest')

                        order = (8, 0, 1)
                        model  = ARIMA(rest_ss, order, freq='D')
                        model = model.fit()
                        model.predict(1, 255)
                    except:
                        try:
                            frac_l[j] = 0.14
                            rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.14, return_sorted=False)
                            rest_ss = pd.Series(rest_as, index=d[:split_date_rank], name='artist' + str(j) + 'rest')

                            order = (8, 0, 1)
                            model  = ARIMA(rest_ss, order, freq='D')
                            model = model.fit()
                            model.predict(1, 255)
                        except:
                            try:
                                frac_l[j] = 0.12
                                rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.12, return_sorted=False)
                                rest_ss = pd.Series(rest_as, index=d[:split_date_rank], name='artist' + str(j) + 'rest')

                                order = (8, 0, 1)
                                model  = ARIMA(rest_ss, order, freq='D')
                                model = model.fit()
                                model.predict(1, 255)
                            except:
                                try:
                                    frac_l[j] = 0.1
                                    rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.1, return_sorted=False)
                                    rest_ss = pd.Series(rest_as, index=d[:split_date_rank], name='artist' + str(j) + 'rest')

                                    order = (8, 0, 1)
                                    model  = ARIMA(rest_ss, order, freq='D')
                                    model = model.fit()
                                    model.predict(1, 255)
                                except:
                                    try:
                                        frac_l[j] = 0.08
                                        rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.08, return_sorted=False)
                                        rest_ss = pd.Series(rest_as, index=d[:split_date_rank], name='artist' + str(j) + 'rest')

                                        order = (8, 0, 1)
                                        model  = ARIMA(rest_ss, order, freq='D')
                                        model = model.fit()
                                        model.predict(1, 255)
                                    except:
                                        frac_l[j] = 0.05
                                        rest_as = sm.nonparametric.lowess(rest, np.asarray(rest_x), frac=0.05, return_sorted=False)
                                        rest_ss = pd.Series(rest_as, index=d[:split_date_rank], name='artist' + str(j) + 'rest')

                                        order = (8, 0, 1)
                                        model  = ARIMA(rest_ss, order, freq='D')
                                        model = model.fit()
                                        model.predict(1, 255)

        rest_pred = model.predict(1, split_date_rank+61)
        rest_pred_nda = rest_pred.values

        for i in range(0, 8):
            stl_w_se = np.append(stl_w_se, w_s)
        stl_w_se = np.append(stl_w_se, w_s[:5])

        stl_m_se = np.append(stl_m_se, 0)
        stl_m_se = np.append(stl_m_se, m_s)
        stl_m_se = np.append(stl_m_se, m_s)

        compose_stl = stl_w_se + stl_m_se + rest_pred_nda
        fit_ap = np.exp(compose_stl)
        
        s = pd.Series(fit_ap, index=d2[:split_date_rank + 61], name='artist' + str(j) + 'compose')
        diff_rate = (abs(fit_ap-orig_arr)/orig_arr)**2
        sigma = math.sqrt(diff_rate.sum()/len(fit_ap))
        phi = math.sqrt(orig_arr.sum()) 
        F.append((1 - sigma) * phi)
    
    result = 0
    for i in range(0, len(F)):
        result += F[i]
    print 'the value of F:', result
    fp.close()
    
'''     
        plt.figure(j)
        s.plot()
        fig = plt.figure(j)
        fig.savefig('pic_trend_3/' + str(j) + '.png')

        artist_id = artists_rank2id[j]
        for idx in range(122, 183):
            date = rank2date[idx]
            play_num = int(math.ceil(fit_ap[idx]))
            if play_num < 0:
                play_num = 0
            row = [artist_id, play_num, date]
            #print row
            fpwriter.writerow(row)
'''
    
if __name__ == '__main__':
    ap_df = APtoDF()
    predstl()
