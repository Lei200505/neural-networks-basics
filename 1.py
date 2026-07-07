import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y_train = train_data['label'].values
X_train = train_data.drop(columns=['label']).values/255

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

num_epochs = 150
lr = 1.5
weight = np.random.rand(784, 10)
one_hot_labels = one_hot(y_train)
for i in range(num_epochs):
  logits = np.dot(X_train, weight)
  predicted = softmax(logits)
  loss = crossentropyloss(predicted, one_hot_labels)
  if i % 5 == 0:
    print(f"Loss in epoch {i}: {loss}")
  weight = gradient_update(predicted, weight, one_hot_labels, X_train, lr)
print(f" {get_accuracy(X_train, weight, y_train)}/{len(X_train)} ")


