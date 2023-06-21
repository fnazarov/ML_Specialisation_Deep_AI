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

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + m
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    return dj_dw/m, dj_db/m

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):

    J_history = []
    p_history = []
    w = w_in
    b= b_in

    for i in range(num_iters):

        dw, db = gradient_function(x, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db

        if i <100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w,b])

        #Print cost every at intervals 10 times or as many iterations if i<10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]}:0.2e} ",
                  f"dw: {dw:0.3e}, db: {db: 0.3e} ",
                  f"w:{w: 0.3e}, b: {b:0.5e}")
    return w, b, J_history, p_history

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730])
compute_cost(x_train, y_train, 1, 2)