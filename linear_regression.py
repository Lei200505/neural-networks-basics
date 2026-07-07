import numpy as np
import matplotlib.pyplot as plt

eps = 10

def create_regression_examples(w_true, b_true, num_examples):
  x = np.linspace(0, 20, num_examples)
  noise = np.random.normal(0, 1, num_examples) * eps
  y = w_true * x + b_true + noise
  return x, y

def plot_regression(x, y, w, b):
  fig, ax = plt.subplots()
  ax.scatter(x, y, c = 'blue', label = "Data points")
  ax.plot(x, w * x + b, c = 'red', label = "Approximation")
  plt.show()

def predict(x, w, b):
  return w * x + b

def update_params(x, y, w, b, lr):
  predicted = predict(x, w, b)
  dw = np.sum((predicted - y) * x) / len(x)
  db = np.sum(predicted - y) / len(x)
  w = w - lr * dw
  b = b - lr * db
  return w, b

x_train, y_train = create_regression_examples(5, 1, 50)
lr = 0.001
w = np.random.rand()
b = np.random.rand()
print(w)
print(b)



for i in range(30):
  w, b = update_params(x_train, y_train, w, b, lr)
  if i % 5 == 0:
    plot_regression(x_train, y_train, w, b)

plot_regression(x_train, y_train, w, b)