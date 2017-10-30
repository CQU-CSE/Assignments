## K nearest neighbor algorithm learn and exercise

#  Instructions
#  ------------
#
#  KNN is a based algorithm for classification and regression problems
#  it actually can be divided into two steps i think:
#  
#  the first step is to construct or build our kd tree which can support
#  us to search the k nearest neighbors quickly rather than linear scanning
#
#  the second step is to search the kd tree and get our hypothesis result
#  with majority voting rule, which means that among our k nearest neighbors
#  if most of them is apple, namely our hypothesis result is apple as well
#
#  this was really a baby step...it's not very easy for the first time i think...
#  wangurns
#  2017-10-30 22:21:06
#


#coding:utf8
import sys
import numpy as np

answer = []
trainSet = []
testSet = []
#K is the number of nearest neighbors
K = 5
#labels of the samples in the training set                   
labels={'Iris-setosa':'0','Iris-versicolor':'1','Iris-virginica':'2'};
     
class Node:
	def _init_(self,val=[],left=None,right=None):
		self.val=val;
		self.left=left;
		self.right=right;

def constructKDTree(X,l):
	if X.size==0:
		return;
	root=Node();
	dimension=l%5 if l!=4 else (l+1)%5;
	column=X[:,dimension];
	X=X[column.argsort()];
	middle=column.size/2;
	root.val=X[middle];
	root.left=constructKDTree(X[:middle],l+1);
	root.right=constructKDTree(X[middle+1:],l+1);
	return root;

def updateCurrentNearestNode(curNode,target,neighbors):
	curDistance=sum((curNode[:-1]-target)**2)**0.5;
	global K;
	if K>0:
		neighbors.append([curDistance,curNode[-1]]);
		K-=1;
	else:
		neighbors.sort();
		if curDistance<neighbors[-1][0]:
			neighbors[-1]=[curDistance,curNode[-1]];

def findKNN(kdTree,target,l,neighbors):
	if kdTree is None:
		return;
	dimension=l%5 if l!=4 else (l+1)%5;
	value=target[dimension];
	nextStepIsLeft=False;
	if value<kdTree.val[dimension]:
		findKNN(kdTree.left,target,l+1,neighbors);
		nextStepIsLeft=True;
	else:
		findKNN(kdTree.right,target,l+1,neighbors);
	updateCurrentNearestNode(kdTree.val,target,neighbors);
	brotherArea=kdTree.right if nextStepIsLeft else kdTree.left;
	if brotherArea is not None:
		d=abs(value-kdTree.val[dimension]);
		neighbors.sort();
		if d < neighbors[-1][0]:
			findKNN(brotherArea,target,l+1,neighbors);

def majorityVoting(neighbors):
	dic=dict();
	for nei in neighbors:
		dic[nei[1]]=dic.get(nei[1],0)+1;
	maxV=None;classK=None;
	for k,v in dic.items():
		if maxV is None or v>maxV:
			classK=k;maxV=v;
	return classK;
	
def classify():
	## ====================Part 1: Data Preprocessing====================
	#  Load the data and transform the form of data to what we need. i.e 
    	#  vector or matrix
    	#training set and test set input
    	with open('./training.txt') as f:
        	for line in f:
           		row=line.rstrip().split(',');
			row[4]=labels[row[4]];
			global trainSet;
			trainSet.append(row);
	trainSet=np.array(trainSet).astype(float);
	with open('./test.txt') as f:
		for line in f:
			row=line.rstrip().split(',');
			testSet.append(row);
	Xtest=np.array(testSet).astype(float);


	## ====================Part 2: Construct KD Tree====================
    	#  Construct the k nearest neighbor tree with our training set 
    	#
	#Construct KD Tree
	kdTree=constructKDTree(trainSet,0);


	## ====================Part 3: Prediction By knn====================
    	#   Predict the result via k nearest neighbors algorithm
    	#   which will fist find k nearest neighbors via our kdTree and then 
	#   make decision via majority voting rule
	global K;
	K_backup=K;
	for xtest in Xtest:
		neighbors=list();
		findKNN(kdTree,xtest,0,neighbors);
		K=K_backup;
		hypothesis=majorityVoting(neighbors);
		print xtest,hypothesis;

if __name__ == '__main__':
    classify()
