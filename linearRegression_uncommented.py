import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
height = pd.read_csv('data.csv', usecols = [0])
weight = pd.read_csv('data.csv', usecols = [1])
x = np.squeeze(np.array(height))
y = np.squeeze(np.array(weight))
x_training_data = x[:-200].reshape(1800,1)
x_test_data = x[-200:].reshape(200,1)
y_training_data = y[:-200]
y_test_data = y[-200:]
regr = linear_model.LinearRegression()
regr.fit(x_training_data, y_training_data)
plt.scatter(x_test_data, y_test_data,  color='black')
plt.plot(x_test_data, regr.predict(x_test_data), color='red',
         linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
