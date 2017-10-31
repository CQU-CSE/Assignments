import numpy as np
import math
class KNN(object):
    data =np.zeros((150,5))
    classone='Iris-setosa\n'
    classtwo='Iris-versicolor\n'
    classthree='Iris-virginica\n'
    i=0
    for line in open("/Users/zhaozehua/Desktop/iris.txt"):
        data[i][0] = line.split(",")[0]
        data[i][1] = line.split(",")[1]
        data[i][2] = line.split(",")[2]
        data[i][3] = line.split(",")[3]
        iristype = line.split(",")[4]
        if iristype == classone:
            data[i][4] = '0'
        elif iristype == classtwo:
            data[i][4] = '1'
        elif iristype == classthree:
            data[i][4] = '2'
        i=i+1

    count = 0
    error = 0
    no = 0
    k = input("k:")
    for count in range(5):
        #trainingset,testset
        num = 0;
        testdata = np.zeros((30,5))
        trainingdata = np.zeros((120,5))
        te = 0
        tr = 0
        for num in range(150):
            if num%5 == count:
                testdata[te] = data[num]
                te = te + 1
            else:
                trainingdata[tr] = data[num]
                tr = tr + 1
        #KNN
        distance = np.zeros((30,120))
        a = 0
        for a in range(30):
            b = 0
            for b in range(120):
                distance[a][b]=(testdata[a][0]-trainingdata[b][0])*(testdata[a][0]-trainingdata[b][0])+(testdata[a][1]-trainingdata[b][1])*(testdata[a][1]-trainingdata[b][1])+(testdata[a][2]-trainingdata[b][2])*(testdata[a][2]-trainingdata[b][2])+(testdata[a][3]-trainingdata[b][3])*(testdata[a][3]-trainingdata[b][3])
                distance[a][b]=math.sqrt(distance[a][b])
        testcount = 0
        for testcount in range(30):
            topk = np.zeros((k, 2))
            kn = 0
            for kn in range(k):
                topk[kn][0]=distance[testcount][kn]
                topk[kn][1]=kn
            for kn in range(k,120):
                d = 0
                for d in range(k):
                    if topk[d][0] > distance[testcount][kn]:
                        topk[d][0] = distance[testcount][kn]
                        topk[d][1] = kn
                        break
            re = 0
            ca = 0
            ca_count = 0
            cb = 0
            cb_count = 0
            cc = 0
            cc_count = 0
            for re in range(k):
                index = int(topk[re][1])
                dis = topk[re][0]
                #print dis
                tag = trainingdata[index][4]
                if tag == 0.0:
                    ca = ca + dis
                    ca_count = ca_count + 1
                elif tag == 1.0:
                    cb = cb + dis
                    cb_count = cb_count + 1
                else:
                    cc = cc + dis
                    cc_count = cc_count + 1
            if ca_count > 0:
                ca = ca_count / ca
            if cb_count > 0:
                cb = cb_count / cb
            if cc_count > 0:
                cc = cc_count / cc
            if cc > ca and cc > cb:
                tag_re = 2
            if cb > ca and cb > cc:
                tag_re = 1
            if ca > cb and ca > cc:
                tag_re = 0
            if tag_re <> testdata[testcount][4]:
                error=error+1
                print "**************"
                print tag_re,testdata[testcount][4]
                print "**************"
            print ca_count,cb_count,cc_count
            print ca,cb,cc
            no = no + 1
            print "################################",no
    error = (150-error) / 150.0

    print error

