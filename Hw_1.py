plt.scatter(X,y, color = 'blue', marker = '*')
plt.grid()
plt.rcParams["figure.figsize"]=(10,10)
plt.xlabel('X1')
plt.ylabel('Y')
plt.title('Dataset for X1 and Y')import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#PART 1
df = pd.read_csv('D3.csv', usecols = ['X1', 'Y'])
df.head()
M = len(df)
X = df.values[:,0]
y = df.values[:,1]
m = len(y)

print('X = ', X[: 100])
print('Y = ', Y[: 100])
print('m = ', m)

X_0 = np.ones((m,1))
X_0[:5]
X_1 = X.reshape(m,1)
X_1[:10]
X = np.hstack((X_0, X_1))

def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    j = 1/(2*m) * np.sum(sqrErrors)
    return j

cost = compute_cost(X, y, theta)
print('The cost for given values of theta_0 and theta_1 =', cost)

def gradient_decent(X, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha/m) * X.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history


theta = [0,0.]
iterations = 1500 
alpha = 0.01

theta, cost_history = gradient_decent(X, y, theta, alpha, iterations)
print('Final value for theta or M and B are:' , theta)
print('Cost function value =' , cost_history)
#Graph of X1 and Y goes here:

plt.scatter(X[:,1],y,color='red',marker='+',label='Training Data')
plt.plot(X[:,1],X.dot(theta),color='green',label='Linear Regression')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('X1 data')
plt.ylabel('Y')
plt.title('Linear Regression Fit')
plt.legend()

theta = [0,0.]
iterations = 1500 
alpha = 0.01

theta, cost_history = gradient_decent(X, y, theta, alpha, iterations)
print('Final value for theta or M and B are:' , theta)
print('Cost function value =' , cost_history)

df = pd.read_csv('D3.csv', usecols = ['X2', 'Y'])
df.head()
X = df.values[:,0]
y = df.values[:,1]
m = len(y)

X_0 = np.ones((m,1))
X_1 = X.reshape(m,1)
X = np.hstack((X_0, X_1))
X[:5]
theta = np.zeros(2)

theta = [0,0.]
iterations = 1500 
alpha = 0.01

theta, cost_history = gradient_decent(X, y, theta, alpha, iterations)
print('Final value for theta or M and B are:' , theta)
print('Cost function value =' , cost_history)

#graph of X2 and Y here:
plt.scatter(X[:,1],y,color='red',marker='+',label='Training Data')
plt.plot(X[:,1],X.dot(theta),color='green',label='Linear Regression')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('X2 data')
plt.ylabel('Y')
plt.title('Linear Regression Fit')
plt.legend()

df = pd.read_csv('D3.csv', usecols = ['X3', 'Y'])

X = df.values[:,0]
y = df.values[:,1]
m = len(y)
plt.scatter(X, y, color = 'blue', marker = '*')



X_0 = np.ones((m,1))
X_1 = X.reshape(m,1)
X = np.hstack((X_0, X_1))
theta = np.zeros(2)

theta = [0,0.]
iterations = 1500 
alpha = 0.01

theta, cost_history = gradient_decent(X, y, theta, alpha, iterations)
print('Final value for theta or M and B are:' , theta)
print('Cost function value =' , cost_history)
plt.scatter(X[:,1],y,color='red',marker='+',label='Training Data')
plt.plot(X[:,1],X.dot(theta),color='green',label='Linear Regression')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('X3 data')
plt.ylabel('Y')
plt.title('Linear Regression Fit')
plt.legend()

