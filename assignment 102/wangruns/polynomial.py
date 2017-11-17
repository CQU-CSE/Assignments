## Polynomial regression algorithm learn and exercise

#  Instructions
#  ------------
#
#  Polynomial regression is a based algorithm for regression problems
#  it actual can be divided into three steps i think:
#
#  the first step is to extend our features, say that we the dimensions of the
#  feature of training set X is n.
#  so the original feature is x1,x2,x3...xn   
#  extended to:
#  x1,x2,x3...xn,
#  x1x1,x1x2,x1x3,...x1xn,
#  x2x2,x2x3...x2xn,
#  ...xnxn
#  even more...
#
#  the second step is to define our polynomial hypothesis function according to 
#  the dimensions of the features extended by us.
#  hypothesis=theta0+theta1*x1+...+thetan*xn+thetam*x1xn+...thetapxnxn...even more
#  
#  the third step is to find the most suitable parameters theta, which fit
#  our trainig set and cross validation set well. To make this come true, we
#  will use gradient descent to get proper parameters in this exercise.
#
#  this was really a baby step...it's not very easy for the first time i think...
#  wangurns
#  2017-11-17 15:49:29
#


#coding:utf8

import sys
import numpy as np
import pandas as pd

answer = []
trainSet = []
testSet = []
#the path of train.txt and test.txt files, here current path is used
path='';
#the number of iterations of gradient descent
iteration=500;
#the learning rate for each iteration
alpha=0.03;
#lambda is the parameter to trade off between overfiting and underfitting
lambdas=[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30];
#the degree of power in the polynomial that cann't be too large otherwise
#it'll overflow and you will get 'NaN', so here i set it to 3. Actually,you
#may notice that if we set it to 1, and it'll change to  a linear problem.
degree=3;

def estimateWithRegularizedTerm(theta,X_val,y_val,lamba):
	m=len(y_val);
	X=np.c_[np.ones(m),X_val];
	y_pre=X.dot(theta);
	error=y_pre-y_val;
	#J=1./(2*m)*sum(error**2);
	J=1./(2*m)*sum(error**2) + lamba/(2*m)*(sum(theta[1:]**2));
	print 'the squared error J for validation set is',float(J);
	return J;

def featureNormalize(X_train):
	mu=X_train.mean(axis=0);
	sigma=np.std(X_train,axis=0);
	X_norm=(X_train-mu)/sigma;
	return X_norm;

"""
featureExtend(X,n,degree) is a function that return the extended features of the degree of power.
say that we have 5 features, x1, x2 x3, x4, x5 originally and we set the degree of power 
to 1,then what then function does is to return the extended features which may look like 
as followed:
x1, x2, x3, x4, x5

say that we set the degree of power to 2, and it may return as followed:
x1x1, x1x2, x1x3, x1x4, x1x5
x2x2, x2x3, x2x4, x2x5
x3x3, x3x4, x3x5
x4x4, x4x5
x5x5

notice that the relationship between degree 1 and degree 2. now you should find that:
degree 1 is: x1, x2, x3, x4, x5
degree 2 seems to be each of the original features multiplying degree 1 separately.
all the original features:
x1	it seems that	x1*degree1	which is	x1x1, x1x2, x1x3, x1x4, x1x5
x2	it seems that	x2*degree1	which is	x2x2, x2x3, x2x4, x2x5	(x2x1 removed)
x3	it seems that	x3*degree1	which is	x3x3, x3x4, x3x5
x4	it seems that	x4*degree1	which is	x4x4, x4x5
x5	it seems that	x5*degree1	which is	x5x5

namely, if we konw the degree 1, we can also calcalute the degree 2 by the relations above
luckily, we have already know the degree 1, which is our original X. Actually, this is the
intuition of the algorithm in this function.

but there is still a problem to overcome, which is the 'removed ones above', you must know
which one should be removed so that there aren't duplicated ones. In the code i will use a
couple of flags to record  the location, where the original features firstly appearing in
the last degree. more specially:
x1's first apprear location in degree1 is 0, x1 * degree1 will start from 0, that is to say:
x1*[x1,x2,x3,x4,x5]	=	x1x1, x1x2, x1x3, x1x4, x1x5
and x2's first appear location in degree1 is 1, so x2*degree1 will start from 1:
x2*[x2,x3,x4,x5]	=	x2x2, x2x3, x2x4, x2x5
...
and x5's first appear location in degree1 is 4, so x5*degree1 will start from 4:
x5*[x5] 	=		x5x5
"""
def featureExtend(X,n,degree):
	if degree == 1:
		lastFirstAppearLocation=[i for i in range(n)];
		return X,lastFirstAppearLocation,X.copy();
	X_Last_Extended,lastFirstAppearLocation,X_extended=featureExtend(X,n,degree-1);
	X_current_Extended=None;
	#each of the original features
	for cInX in range(n):
		xColumn=X[:,cInX]; 
		#mark whether the original feature is first appearing
		isFirstAppear=True;
		for cInLastExtendedX in range(lastFirstAppearLocation[cInX],X_Last_Extended.shape[1]):
			if isFirstAppear:
				#update lastFirstAppearLocation
				if X_current_Extended is None:
					lastFirstAppearLocation[cInX]=cInLastExtendedX;
				else:
					lastFirstAppearLocation[cInX]=X_current_Extended.shape[1];
				isFirstAppear=False;
			newFeatureColumn=xColumn*X_Last_Extended[:,cInLastExtendedX];
			if X_current_Extended is not None:
				X_current_Extended=np.c_[X_current_Extended,newFeatureColumn];
			else:
				X_current_Extended=newFeatureColumn;
	#add all the extended features together
	X_extended=np.c_[X_extended,X_current_Extended];
	return X_current_Extended,lastFirstAppearLocation,X_extended;
			
def polynominalRegression(X,y,alpha,iteration,lamba,degree):
	m=len(y);
	X=np.c_[np.ones(m),X];
	theta=np.random.rand(len(X[0]));
	for i in range(iteration):
		#gradientDescent with regularized term
		error=X.dot(theta)-y;
		thetaWithoutTheta0=theta.copy();
		thetaWithoutTheta0[0]=0;
		"""
		#the cost J is reducing after each iteration totally
		J=1./(2*m)*sum(error**2) +   lamba/(2*m)*(sum(thetaWithoutTheta0**2));
		print 'the squared cost J is',float(J);
		"""
		theta-=alpha*( 1.0/m*(X.T.dot(error)) +   lamba/m*thetaWithoutTheta0 );
	return theta;

def train():
	## ====================Part 1: Data Preprocessing====================
	#  Load the data and transform the form of data to what we need. e.g
	#  vector or matrix
	#training set and test set input
	df=pd.read_table(path+'./noise_train.txt',header=None,sep=',');
	df_test=pd.read_table(path+'./noise_test.txt',header=None,sep=',');


	## ===============Part 2: Choose data and separate label===============
	#  randomly choose training set and cross validation set, here i set the
	#  ratio to 7:3
	#  then separate the features and label
	#randomly choose training set and cross validation set
	ratio=0.7;
	labelIndex=5;
	n=int(ratio*len(df));
	trainSet=df[:n].sample(n);
	validationSet=df[n:].sample(len(df)-n);
	X_train=trainSet.drop(labelIndex,axis=1).values;
	y_train=trainSet[labelIndex].values;
	X_val=validationSet.drop(labelIndex,axis=1).values;
	y_val=validationSet[labelIndex].values;
	X_test=df_test.values;
	#feature extend
	#degree=3;
	n=X_train.shape[1];
	_1,_2,X_train_extended=featureExtend(X_train,n,degree);
	_1,_2,X_val_extended=featureExtend(X_val,n,degree);
	_1,_2,X_test_extended=featureExtend(X_test,n,degree);
	#feature normalize
	X_train_extended_norm=featureNormalize(X_train_extended);
	X_val_extended_norm=featureNormalize(X_val_extended);
	X_test_extended_norm=featureNormalize(X_test_extended);


	## ====================Part 3: Training====================
	#  notice that we have 5 features in our samples, but we will
	#  extend them, here i choose the degree of power is 2, you 
	#  can also choose others.
	#  h=theta0 + theta1*x1 + theta2*x2 + ... + theta5*x5 +
	#          theta11*x1x1 + theta12*x1x2 + ... + theta15*x1x5 +
	#          ... + theta51*x5x5
	#  as you know, our task is to find the most suitable parameters
	#  theta here to fit our training set and cross validation set
	#  
	lamba=1.0;
	theta=polynominalRegression(X_train_extended_norm,y_train,alpha,iteration,lamba,degree);
	
	
	## ====================Part 4: Cross Validation====================
	#  after we get our hypothesis model via our training set, then
	#  we need to use cross validation to see how well it works for
	#  the new samples.
	#  we can see that the squared error J of all the cross validation
	#  data set is about 18.2061851842
	#estimate
	J=estimateWithRegularizedTerm(theta,X_val_extended_norm,y_val,lamba);
	print 'Please any key to begin validating...';
	sys.stdin.readline();
	# choose best lambda according to the result of cross validation
	for lam in lambdas:
		theta_temp=polynominalRegression(X_train_extended_norm,y_train,alpha,iteration,lam,degree);
		J_temp=estimateWithRegularizedTerm(theta,X_val_extended_norm,y_val,lam);
		print 'lambda is ',lam,' and the cost J is ',J_temp;
		if J_temp<J:
			J=J_temp;theta=theta_temp;lamba=lam;
	print '\nthe best lambda is ',lamba,' and the cost J is ',J;
	print 'Please any key to begin predicting...';
	sys.stdin.readline();

	## ====================Part 5: Prediction====================
	#  use the best parameters we got above to predict
	#
	#test
	X_test_extended_norm=np.c_[np.ones(len(X_test)),X_test_extended_norm];
	y_pre=X_test_extended_norm.dot(theta);
	cnt=0;
	for i in X_test:
		print i,'predicted as>>',y_pre[cnt];cnt+=1;
	
	
if __name__ == '__main__':
	train()

