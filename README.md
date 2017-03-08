# MLTutorial
Made for the IEEE Ryerson Machine Learning Tutorial for Beginners 
https://docs.google.com/presentation/d/1QcgoomUi-vGVbkRTQncp55VtrM-YBvYZFHgQ1XZk76I/edit?usp=sharing





```
#This was put together by Anthony Sistilli and was mostly copied from http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
#Height is in inches and is the 1st column ([0]) in the datas
height = pd.read_csv('data.csv', usecols = [0])
#Weight is in lbs and is the 2nd column ([1]) in the data
weight = pd.read_csv('data.csv', usecols = [1])

#Formatting the data into readable form
x = np.squeeze(np.array(height))
y = np.squeeze(np.array(weight))

#To print a scatter plot of your data, uncomment the next two lines
plt.scatter(x, y)
plt.show()

#splitting the data up into training and test sets
#split:
#train = 1800
#test = 200
#Note that the .reshape is to get rid of a numpy to scikit array size error
x_training_data = x[:-200].reshape(1800,1)
x_test_data = x[-200:].reshape(200,1)

y_training_data = y[:-200]
y_test_data = y[-200:]

#Creating the linear regression model object
regr = linear_model.LinearRegression()

#Training your model on the training data
regr.fit(x_training_data, y_training_data)

# Plotting the output
plt.scatter(x_test_data, y_test_data,  color='black')
plt.plot(x_test_data, regr.predict(x_test_data), color='red',
         linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

```
