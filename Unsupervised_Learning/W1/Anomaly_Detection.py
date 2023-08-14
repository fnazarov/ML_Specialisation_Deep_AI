import numpy as np
import matplotlib.pyplot as plt
from utils import *


#Load the dataset
X_train, X_val, y_val = load_data()

# Display the first five elements of X_train
print("The first 5 elements of X_train are:\n", X_train[:5])

# Display the first five elements of X_val
print("The first 5 elements of X_val are\n", X_val[:5])

# Display the first five elements of y_val
print("The first 5 elements of y_val are\n", y_val[:5])
#Check the dimensions of your variables
print ('The shape of X_train is:', X_train.shape)
print ('The shape of X_val is:', X_val.shape)
print ('The shape of y_val is: ', y_val.shape)

# Create a scatter plot of the data. To change the markers to blue "x",
# we used the 'marker' and 'c' parameters
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b')

# Set the title
plt.title("The first dataset")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latency (ms)')
# Set axis range
plt.axis([0, 30, 0, 30])
plt.show()


def estimate_gaussian(X):
    """
    Calculates mean and variance of all features
    in the dataset

    Args:
        X (ndarray): (m, n) Data matrix

    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m,n = X.shape
    mu = 0

    ### START CODE HERE ###
    for i in range ( m ):
        mu += X[ i ]
    mu = mu / m

    var = 0
    for i in range ( m ):
        var += ((X[ i ] - mu) ** 2)
    var = var / m

    ### END CODE HERE ###

    return mu,var

mu, var = estimate_gaussian(X_train)
print("Mean of each feature:", mu)
print("Variance of each feature:", var)

def multivariate_gaussian(X, mu, var):
    """
     Computes the probability
     density function of the examples X under the multivariate gaussian
     distribution with parameters mu and var. If var is a matrix, it is
     treated as the covariance matrix. If var is a vector, it is treated
     as the var values of the variances in each dimension (a diagonal
     covariance matrix

     """
    k = len(mu)

    if var.ndim == 1:
        var = np.diag(var)

    X = X -mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5)*\
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) *X , axis = 1))
    return p
def select_threshold(y_val,p_val):

    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val))/1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        anom = (p_val < epsilon)
        tp = sum((a and b) for a, b in zip(anom, y_val))
        fp = sum((a and not b) for a,b in zip(anom, y_val))
        fn = sum((not a and b) for a,b in zip(anom, y_val))

        prec = tp/(tp + fp)
        rec = tp/(tp + fn)
        F1 = (2*prec*rec) / (prec + rec)

        if F1>best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1


p = multivariate_gaussian(X_train, mu, var)

p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

#Find the outliers in the training set
outliers = p < epsilon
#Visualize the fit
visualize_fit(X_train, mu, var)
# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)

###High dimensional dataset¶

#Now, we will run the anomaly detection algorithm that you implemented on a more realistic and much harder dataset.
X_train_high, X_val_high, y_val_high = load_data_multi()
print ('The shape of X_train_high is:', X_train_high.shape)
print ('The shape of X_val_high is:', X_val_high.shape)
print ('The shape of y_val_high is: ', y_val_high.shape)

# Apply the same steps to the larger dataset

# Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)
# Evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)
# Evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)
# Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))