#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Just wrote the function for the cost function although it is not called
# as the cost function is incorporated in the gradient descent function.

# def cost_function(a, b, X, Y):
#     m = len(X)    # size of the dataset
#     hypothesis = 0
#     for i in range(m):
#         x = X[i]   # this is int
#         y = Y[i]   # this is float
#         hypothesis += (y - (a*x + b))**2
    
#     hypothesis = hypothesis/(2*m)
    


#%%

def gradient_descent(theta0_now, theta1_now, alpha, X, Y):
    """this function minimizes the cost function and find out the
    optimal parameters for the dataset

    Args:
        theta0_now (float): first parameter
        theta1_now (float): second parameter
        alpha (float): learning rate of the algorithm
        X (pandas.series): feature of the dataset
        Y (pandas.series): labeled output of the dataset

    Returns:
        tuple: calculated parameters after training
    """
    m = len(X)
    theta0_gradient = 0
    theta1_gradient = 0
    for i in range(m):
        x = X[i] 
        y = Y[i]
        
        theta0_gradient += (theta0_now*x + theta1_now - y) * x * (1/m)
        theta1_gradient += (theta0_now*x + theta1_now - y) * (1/m)
    
    theta0_now = theta0_now - alpha * theta0_gradient
    theta1_now = theta1_now - alpha * theta1_gradient
    
    return theta0_now, theta1_now


# %%
def train_model(X, y, epochs=500):
    """this function trains the model on the dataset by calling the gradient descent function.

    Args:
        X (pandas.series): feature of the dataset
        y (pandas.series): labeled output of the dataset
        epochs (int, optional): Number of iterations to train the dataset. Defaults to 500.

    Returns:
        tuple: calculated parameters after training
    """
    theta0 = 1.0
    theta1 = 2.0
    alpha = 0.0001
    for i in range(epochs):
        if i%50 == 0:
            print(f"Epoch = {i}")
        theta0, theta1 = gradient_descent(theta0, theta1, alpha, X, y)
    print("Training of the dataset is complete!")
    return theta0, theta1

#%%

data = pd.read_csv("random_dataset.csv")    # reading dataset

column_X = data.columns[0]
column_Y = data.columns[1]
X = data[column_X]            # input value
y = data[column_Y]            # output value

parameter_list = train_model(X, y)

# %%
# to visualize the linear function and the data points.
# just uncomment the code.

# theta0 = parameter_list[0]
# theta1 = parameter_list[1]
# column_data = data.columns[0]
# column_res = data.columns[1]
# X = data[column_data]
# y = data[column_res]
# plt.scatter(X, y)
# plt.plot(X, [theta0*x + theta1 for x in X], color='red')

# %%
