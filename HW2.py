import numpy as np
import pandas as pd
# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

housing=pd.DataFrame(pd.read_csv("Housing.csv"))
housing.head()

varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea',]
def binary_map(x):
    return x.map({'yes': 1, 'no': 0})

housing[varlist] = housing[varlist].apply(binary_map)

from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = .7, test_size = .3,random_state = 0)
df_train

num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]
df_Newtrain.head()

num_varsb = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom','basement','hotwaterheating', 'airconditioning', 'parking', 'price']
df_Newtrainb = df_train[num_varsb]
df_Newtestb = df_test[num_varsb]

df_Newtrain

df_Newtestb

y_Newtrain = df_Newtrain.pop('price')
x_Newtrain = df_Newtrain

y_newTest = df_Newtest.pop('price')
x_newTest = df_Newtest

x_newTraintB = df_Newtrainb
y_newTraintB = df_Newtrainb.pop('price')

x_newTestb = df_Newtestb
y_newTestb = df_Newtestb.pop('price')


x_train = x_Newtrain.to_numpy()
y_train = y_Newtrain.to_numpy()

#x_train = np.c_[np.ones(len(x_train),dtype='int64'),x_train]


x_test = x_newTest.to_numpy()
#X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]
y_test = y_newTest.to_numpy()


def LinReg_with_gradient_descent(X, y, alpha, iterations):
    m = X.shape[0]  # number of samples
    ones =np.ones((m,1))  #creating values for ones with m lenght and 1 values
    X = np.concatenate((ones, X), axis=1)   # concating the ones matrix with X so they are joined together, this helpes for theta 0 to be multiplied by 1
    n = X.shape[1]   # 
    Theta = np.ones(n)     #parameter initialization of all ones for theta
    h = np.dot(X, Theta)   # Compute hypothesis

  # Gradient descent algorithm
    cost = np.ones(iterations)
    for i in range (0, iterations):
        Theta[0] = Theta[0] - (alpha / m) * sum(h-y)
        for j in range(1, n):
            Theta[j]= Theta[j] - (alpha/ m) * sum((h-y) * X[:, j])
        h  = np.dot(X, Theta)
        cost[i] = 1/(2*m) * sum(np.square(h-y))  # Compute Cost
    return cost, Theta

cost, theta = LinReg_with_gradient_descent(x_train, y_train, 0.00000000001, 10000)
plt.plot(cost) #Cost function graph for 1a
plt.xlabel("number of iteration 1 a")
plt.ylabel( "Loss Function")
plt.show()
plt.plot(costb)  #Cost function graph for 1b
plt.xlabel("number of iteration")
plt.ylabel( "Loss Function")
plt.show()

import warnings                                                 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler 

scaler = MinMaxScaler()
df_Newtrain[num_vars] = scaler.fit_transform(df_train[num_vars]) #Normalizing the data
df_Newtrainb[num_varsb] = scaler.fit_transform(df_train[num_varsb])

def LinReg_with_gdwithpen(X, y, alpha, lbda, iterations):
    m = X.shape[0]  # number of samples
    ones =np.ones((m,1))  #creating values for ones with m lenght and 1
    X = np.concatenate((ones, X), axis=1)   # concating the ones matrix with X
    n = X.shape[1]   # 
    Theta = np.ones(n)    #parameter initialization of all ones for theta
    h = np.dot(X, Theta)   # Compute hypothesis

  # Gradient descent algorithm
    cost = np.ones(iterations)
    for i in range (0, iterations):
        Theta[0] = Theta[0] - (alpha / m) * sum(h-y)
        for j in range(1, n):
            Theta[j]= Theta[j]*(1-alpha*(lbda)/(m)) - (alpha/ m) * sum((h-y) * X[:, j])
        h  = np.dot(X, Theta)
        cost[i] = 1/(2*m) * sum(np.square(h-y))  # Compute Cost
    return cost, Theta
    
    costg, thetag = LinReg_with_gdwithpen(x_newTest, ynormal, 0.001, 10, 1000)





