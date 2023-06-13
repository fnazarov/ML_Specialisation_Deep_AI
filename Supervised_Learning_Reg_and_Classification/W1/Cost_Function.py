import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x, y, w, b):
    m = x.shape[ 0 ]
    f_x = np.zeros(m)
    cost_sum = 0
    for j in range(m):
        f_x = w * x[j] + b
        cost = (f_x - y[j])**2
        cost_sum = cost_sum + cost

    J = 1/(2*m) * cost_sum

    return J


x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730])
compute_cost(x_train, y_train, 1, 2)