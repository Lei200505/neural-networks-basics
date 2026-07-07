import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

num_of_features=2
num_of_centers=4
inputs, labels = make_blobs(n_samples = 150, n_features = num_of_features, centers = num_of_centers, random_state =6, cluster_std = 1.3)
plt.scatter(inputs[:,0],inputs[:,1],c=labels)
plt.title("Dataset")
plt.show()

#One-hot elkódolás
def one_hot(y):
  shape = (y.size, int(np.max(y)) + 1)
  rows = np.arange(y.size)
  new_y = np.zeros(shape)
  new_y[rows, y] = 1
  return new_y

def softmax(x):
  exps = np.exp(x)
  exps_sum = np.sum(exps, axis = 1, keepdims = True)
  return exps / exps_sum

def crossentropyloss(predicted, one_hot_labels):
  loss = -np.mean(np.sum(np.log(predicted) * one_hot_labels, axis = 1))
  return loss
def gradient_update(predicted, weight, one_hot_labels, data, lr):
  num_datapoints = np.shape(data)[0]
  gradient = 1 / num_datapoints * data.T.dot(predicted - one_hot_labels)
  weight = weight - lr * gradient
  return weight
def get_accuracy(inputs, weight, labels):
  predicted = softmax(np.dot(inputs, weight))
  predicted_index = np.argmax(predicted, axis = 1)
  return np.sum(predicted_index == labels)

#Betanítás
num_epochs = 20
lr = 0.01
weight = np.random.rand(num_of_features, num_of_centers)
one_hot_labels = one_hot(labels)
for i in range(num_epochs):
  logits = np.dot(inputs, weight)
  predicted = softmax(logits)
  loss = crossentropyloss(predicted, one_hot_labels)
  print(f"Loss in epoch {i}: {loss}")
  weight = gradient_update(predicted, weight, one_hot_labels, inputs, lr)

print(f" {get_accuracy(inputs, weight, labels)}/ {len(inputs)}")


