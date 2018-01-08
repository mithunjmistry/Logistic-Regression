
# coding: utf-8

# In[83]:

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[84]:

iris_raw = datasets.load_iris()


# In[85]:

x = iris_raw.data[:,-2:][50:]


# In[86]:

y = iris_raw.target[50:]
y[y == 1] = 0
y[y == 2] = 1


# ### Mean center the data

# In[87]:

x = ((x - x.min(axis=0))/(x.max(axis=0) - x.min(axis=0)))


# In[88]:

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[105]:

def logistic_regression(x, y):
    iteration = 200
    n = x.shape[1]
    m = x.shape[0]
    #add bias term
    x = np.hstack((np.ones((m,1)),x))
    alpha = 5
    error_total = 0 
    #initialize theta
    theta = np.zeros(3)
    theta = np.matrix(theta)
    
    for skip in range(0,100):
        # flush theta each time
        theta = np.zeros(3)
        theta = np.matrix(theta)
    
        x_test = x[skip]
        x_input = np.delete(x, (skip), axis=0)
        y_test = y[skip]
        y_output = np.delete(y, (skip), axis=0)

        for index_iter in range(iteration-1):
            cur_z = np.dot(x_input,theta.T)
            cur_y_hat = sigmoid(cur_z)
            cur_residual = cur_y_hat - np.array([y_output]).T

            p_d = np.zeros(3)
            
            for i in range(0, 3):
                d = np.dot(cur_residual.T, (x_input[:, i]))
                p_d[i] = (1.0 / x_input.shape[0]) * d * - 1

            #update theta
            theta += alpha*p_d
            
        # testing
        temp = np.dot(x_test,theta.T)
        y_prediction = sigmoid(temp)
        if(y_prediction >= 0.5):
            y_prediction = 1
        elif(y_prediction < 0.5):
            y_prediction = 0
        error = np.abs(y_prediction - y_test)
        error_total += error
    return error_total/m


# In[108]:

avg_error = logistic_regression(x, y)


# In[109]:

print("Average error for Logistic Regression is {}".format(avg_error))


# In[ ]:



