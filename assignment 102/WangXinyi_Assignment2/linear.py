#coding:utf8
import re
import sys
import numpy as np
import random
outArray=[]
answer = []
trainSet = []
testSet = []
trainLabel=[]
testPredict=[]
regularationRate=0.01  #正则化因子
learningRate=0.9
error=np.zeros(1052)
max1=min1=max2=min2=max3=min3=max4=min4=max5=min5=0
'''
学习率,模型维度为1,正则化因子取1的的时候，学习率取0.00000011模型效果最好，此时拟合程度不佳，220万次迭代后损失收敛为605，
，将每次迭代变化的收敛尺度设为0.01学习率不变，正则化因子1，迭代14433次收敛至676.625，正则化因子0.1，迭代14423次收敛至
674.399，正则化因子0.01时14422次迭代后收敛至674.221.当模型维度取2，正则化因子取1时，如果还是把每次迭代变化的收敛尺度
设为0.01则很快就收敛了，此时变为0.00001，正则化因子为1经过10万次迭代收敛至6651.358.正则化因子0.1，10万次迭代收敛至6621.548
发生上述现象是因为没有进行特征缩放'''
modelDimension=3         #确定模型维度
parameter=np.random.rand(5) #随机初始化参数θ
parameterArray=[]#将参数存储在列表中
sumDif=0         #定义输出值与真实值之差的和
Epsilon=0.01#梯度下降收敛尺度。当损失函数值小于Epsilon时即收敛
sumParameter=0#正则化项，参数之和

def training():
    global parameterArray
	#training
    with open('./noise_train.txt') as f:
        for line in f:
           #you can add your code below
            trainExample=map(float,(line.split(','))[0:5])#将每行分段，保存其中第一个到第5个元素，即训练数据。
            trainlabel=(map(float,(re.split('[,\s]',line))[0:6]))[5]#保存第六个元素，即训练标签。
            trainSet.append(trainExample)
            trainLabel.append(trainlabel)
        #进行特征缩放
        global max1,min1,max2,min2,max3,min3,max4,min4,max5,min5
        max1=max(trainSet,key=lambda tuplex:tuplex[0])[0]
        min1=min(trainSet,key=lambda tuplex:tuplex[0])[0]
        max2=max(trainSet,key=lambda tuplex:tuplex[1])[1]
        min2=min(trainSet,key=lambda tuplex:tuplex[1])[1]
        max3=max(trainSet,key=lambda tuplex:tuplex[2])[2]
        min3=min(trainSet,key=lambda tuplex:tuplex[2])[2]
        max4=max(trainSet,key=lambda tuplex:tuplex[3])[3]
        min4=min(trainSet,key=lambda tuplex:tuplex[3])[3]
        max5=max(trainSet,key=lambda tuplex:tuplex[4])[4]
        min5=min(trainSet,key=lambda tuplex:tuplex[4])[4]
        for i in range(0,len(trainSet)):
            trainSet[i][0]=(trainSet[i][0]-min1)/(max1-min1)
            trainSet[i][1]=(trainSet[i][1]-min2)/(max2-min2)
            trainSet[i][2]=(trainSet[i][2]-min3)/(max3-min3)
            trainSet[i][3]=(trainSet[i][3]-min4)/(max4-min4)
            trainSet[i][4]=(trainSet[i][4]-min5)/(max5-min5)
        exampleNum=len(trainSet)            
        theta0=np.zeros(1)#初始化θ
     
        parameterArray.append(theta0)
        for i in range(1,modelDimension+1): #初始化x,x^2,x^3等的参数θ1，2，3，4，5...
            parameter=np.zeros(5)
            parameterArray.append(parameter) #***将各维的参数保存，注意这里向量存进去之后变成了数列。
        gradientDescent(exampleNum,trainSet,parameterArray)

def testing():
    global testPredict
    #test
    with open('./noise_test.txt') as f:
        for line in f:
            testExample=map(float,(line.split(','))[0:5])
            testSet.append(testExample)
           #you can add your code below
        for i in range(0,len(testSet)):
            testSet[i][0]=(testSet[i][0]-min1)/(max1-min1)
            testSet[i][1]=(testSet[i][1]-min2)/(max2-min2)
            testSet[i][2]=(testSet[i][2]-min3)/(max3-min3)
            testSet[i][3]=(testSet[i][3]-min4)/(max4-min4)
            testSet[i][4]=(testSet[i][4]-min5)/(max5-min5)
        for i in range(0,len(testSet)):
            x=np.array(testSet[i])
            testResult=func(x,parameterArray)
            testPredict.append(testResult)
        testError=0
    with open('./testLabel.txt') as f:
        Label=f.read()       
        testLabel=map(float,Label.split(','))
        for i in range(0,len(testPredict)):
            testError=testError+(testPredict[i]-testLabel[i])**2
        print testError/(2*len(testPredict))
            
def func(x,parameterArray):#描述函数
    sumfunc=parameterArray[0][0]#初始化sumfunc=θ0

    for i in range(1,modelDimension+1): #初始化x,x^2,x^3等的参数θ1，2，3，4，5...
        parameter=parameterArray[i]#将数列转化成向量
        sumfunc=sumfunc+parameter.dot(x**i)     #h(θ)x=θ0+θ1*X+θ2*X^2+θ3*X^3...
         #将各维的参数保存。  
    return sumfunc

def costFunc(parameterArray,exampleNum):#表示损失函数，输入是样本容量以及数据集
    sumDif=0
    sumParameter=0
    for i in range(0,exampleNum):
        x=np.array(trainSet[i])
        sumDif=sumDif+(func(x,parameterArray)-trainLabel[i])**2#计算函数预测与真实值的差的平方
   
    for k in range(0,len(parameterArray)):        #计算各个参数之和
        parameterNow=parameterArray[k]            #将参数向量依次取出
        sumParameter=sumParameter+parameterNow.dot(parameterNow)       #将参数向量的平方θ^2累加  

    return sumDif/(2*exampleNum)+regularationRate/(2*exampleNum)*sumParameter #返回带正则化项的均方误差函数
    

def gradientDescent(exampleNum,trainSet,parameterArray):#梯度下降
    cost=1           #初始化误差
    gradient=[]
    gradient.append(np.random.rand(1))#随机初始化θ1的梯度
    for i in range(1,modelDimension+1):
        gradient.append(np.zeros(5))#用随机数初始化梯度
    
    iterationTime=1#记录迭代次数
    previousCost=0#记录上一次的损失
    deltaCost=2#记录两次迭代之间的损失变化值
    while(cost>Epsilon and iterationTime<5000 and deltaCost>0.0001):#进行梯度下降迭代,迭代次数上限为5000次，误差小于阈值或者每次梯度下降使误差变化值过小则停止迭代
        #梯度下降过程有问题
        gradientSum=[0]*(modelDimension+1)#初始化用于梯度下降的(hθ(x(i))-(y(i)))x,其中gradientSum[0]=0
        for i in range(1,modelDimension+1):
            sumDifgrad=0
            for j in range(0,exampleNum):
                X=np.array(trainSet[j])
                sumDifgrad=sumDifgrad+(func(X,parameterArray)-trainLabel[j])#求出hθ(x(i))-y(i)的和,用于梯度下降
                gradientSum[i]=gradientSum[i]+(func(X,parameterArray)-trainLabel[j])*(X**i) 
            gradient[i]=learningRate*((1.0/exampleNum)*gradientSum[i]+(float(regularationRate)/exampleNum)*parameterArray[i]) #计算出除了偏置梯度以外的梯度
            
        gradient[0]=learningRate*((1.0/exampleNum)*sumDifgrad+(float(regularationRate)/exampleNum)*parameterArray[0]) #记录偏置的梯度
        print parameterArray

        for k in range(0,modelDimension+1):
            parameterArray[k]=parameterArray[k]-gradient[k]

        cost=costFunc(parameterArray,exampleNum)#算出损失
        if iterationTime==1:
            deltaCost=1#初始化delatacost使第一次迭代时不中止，因为第一次迭代时previousCost=0
        else:
            deltaCost=previousCost-cost
        print 'cost=%f'%cost#输出损失
        print 'iterationNumber=%d'%iterationTime#输出迭代次数
        print '\n'#换行
        iterationTime=iterationTime+1#每次循环使迭代次数加1
        previousCost=cost
        
    #测试训练集hθ(x)-y的值，仅限于模型次数为一次
    '''
    for i in range(0,exampleNum):
        parameterArray1=parameterArray[0][0]
        parameterArray2=parameterArray[1]
        parameterArray2=np.array(parameterArray[1])
        trainSet[i]
        out=parameterArray1+parameterArray2.dot(trainSet[i])
        outArray.append(out)
    global error    
    error=np.array(outArray)-np.array(trainLabel)
    sumGradient=np.zeros(5)
    for i in range(0,exampleNum):
        X=np.array(trainSet[i])
        sumGradient=sumGradient+error[i]*X
    print sumGradient
    print sumGradient/exampleNum
    '''
	
	
if __name__ == '__main__':
    training()
    testing()

