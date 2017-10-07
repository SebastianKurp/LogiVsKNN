import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from scipy import stats
import math
np.set_printoptions(threshold = 50)
np.set_printoptions(precision=4)

#Loads Iris dataset
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

ar_train, ar_test = train_test_split(iris_X, test_size=0.30, random_state=42)
ar_dev, ar_test = train_test_split(ar_test,test_size=0.5, random_state=42)

iris_train_X = np.array(ar_train)
iris_dev_X = np.array(ar_dev)
iris_test_X = np.array(ar_test)

ar_train, ar_test = train_test_split(iris_Y, test_size=0.30, random_state=42)
ar_dev, ar_test = train_test_split(ar_test,test_size=0.5, random_state=42)

iris_train_Y = np.array(ar_train)
iris_dev_Y = np.array(ar_dev)
iris_test_Y = np.array(ar_test)



#this fx finds dinstance between points.
#length is the size of the tuple


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
def getKNN(trainingSet,trainY,testInstance,k):
    distances = []
    index = []
    length = len(testInstance)-1
    for x in range(len(trainingSet[:,1])):
        dist = euclideanDistance(testInstance,trainingSet[x,:],length)
        distances.append("%.5f" % round(dist,5))
        index.append(x)
    myNump = np.column_stack((index,trainY,distances))
    myNump = myNump[myNump[:,2].argsort()]
    myNump  =np.delete(myNump,0,1)
    myNump  =np.delete(myNump,1,1)
    myNump = myNump[:k,:]
    m = stats.mode(myNump)
    return m[0]

def Main():   
    yTrain = iris_train_Y.tolist()
    predY = []
    for j in range(len(iris_test_X[:,1])):
        predY.append(getKNN(iris_train_X,yTrain,iris_test_X[j,:],7))
    y_true = iris_test_Y
    y_pred = np.array(predY)
    
    print (y_true)
    print (y_pred)
Main()