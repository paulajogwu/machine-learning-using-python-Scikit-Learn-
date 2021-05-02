import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import  neighbors,metrics


import pandas as pd
import numpy as np

data = pd.read_csv('Thousand.csv')


x = data[['MEM STRGTH(MPA)', 'MEM STRESS(MPA)']].values
y = data[['MEM LIM ST NOD']]

# instantiate the LabelEncoder
le = LabelEncoder()
# the data is stored in a csv file for your reference




# tranform the categorical values into numerical values
label_maping= {'T':0,'F':1}
y['MEM LIM ST NOD'] = y['MEM LIM ST NOD'].map(label_maping)
y= np.array(y)


# instantiate LinearRegression
knn = LinearRegression()
# fit to training data
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.1000)
knn.fit(x_train,y_train)

# predict the output for test data
predicted = knn.predict(x_test)

# find the accuracy of predcition using training data
accuracy = knn.score(x_test,predicted)

# print the final results
print("prediction:",predicted)

print("Accuracy:",accuracy)
