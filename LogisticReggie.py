import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.svm import SVC

np.set_printoptions(threshold = 5)

#Loads Iris dataset()
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

#Spilts data between train,test,dev
#This one splits the data as train and test, 70-30
ar_train, ar_test = train_test_split(iris_X, test_size=0.30, random_state=42)
#This line spilts the test data previously split in the last part in half, so 15%.
ar_dev, ar_test = train_test_split(ar_test,test_size=0.5, random_state=32)

iris_train_X = np.array(ar_train)
iris_dev_X = np.array(ar_dev)
iris_test_X = np.array(ar_test)

#spilts the iris_Y
ar_train, ar_test = train_test_split(iris_Y, test_size=0.30, random_state=42)
ar_dev, ar_test = train_test_split(ar_test,test_size=0.5, random_state=32)

iris_train_Y = np.array(ar_train)
iris_dev_Y = np.array(ar_dev)
iris_test_Y = np.array(ar_test)

# Create linear regression object
logreggie = LogisticRegression()

# Train the model using the training sets
logreggie.fit(iris_train_X, iris_train_Y)

# Prints coefficients
print('Coefficients: \n', logreggie.coef_)
# The mean squared error
print("Mean squared error: %.2f"
    % np.mean((logreggie.predict(iris_test_X) - iris_test_Y) ** 2))
#Variance score of 1 is perfect prediction
print('Variance score: %.2f' % logreggie.score(iris_test_X, iris_test_Y))