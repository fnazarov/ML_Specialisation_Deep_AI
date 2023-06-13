import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0,2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = " {x_train})
print(f"y_train = "{y_train})

plt.scatter(x_train, y_train, marker= 'x', c= 'r')

plt.title("Housing Prices")

plt.ylabel("Price (in 100s of dollars)")

plt.xlabel("Size (1000 sqft)")
plt.show()

def computer_model_output(x, w, b):

    m = x.shapes[0]
    f_wb = np.zeros(m)
    for i range(m):
        f_wb[i] = w * x[i] + b
    return f_wb