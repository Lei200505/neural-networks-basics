import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

inputs, labels = make_blobs(n_samples=300, centers=2, n_features=2, random_state=2)
plt.scatter(inputs[:,0],inputs[:,1],c=labels)
plt.title("Dataset")
plt.show()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def logisticloss(predicted, true_labels):
  eps = 1e-15
  predicted = np.clip(predicted, eps, 1 - eps)
  n = len(true_labels)
  loss = - 1/n * np.sum(true_labels * np.log(predicted) + (1 - true_labels) * np.log(1 - predicted))
  return loss

def gradient_update(predicted, weight, labels, data, lr):
  num_datapoints = len(labels)
  gradient = 1 / num_datapoints * np.dot(data.T, predicted - labels)
  weight = weight - lr * gradient
  return weight

def get_accuracy(inputs, weight, labels):
  predicted = sigmoid(np.dot(inputs, weight))
  predicted_labels = (predicted >= 0.5).astype(int)
  return np.sum(predicted_labels == labels)

num_epochs = 100
lr = 0.1
weight = np.random.rand(2, 1)
labels = labels.reshape(-1, 1)
for i in range(num_epochs):
  logits = np.dot(inputs, weight)
  predicted = sigmoid(logits)
  loss = logisticloss(predicted, labels)
  if i % 5 == 0:
    print(f"Loss in epoch {i}: {loss}")
  weight = gradient_update(predicted, weight, labels, inputs, lr)

get_accuracy(inputs, weight, labels)