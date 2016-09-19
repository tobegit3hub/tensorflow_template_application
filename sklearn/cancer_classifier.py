#!/usr/bin/env python

import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

FEATURE_NUMBER = 9

# Read train and test data
with open("../data/cancer_train.csv", "r") as f:
  train_dataset = np.loadtxt(f, delimiter=",")
  train_labels = train_dataset[:, FEATURE_NUMBER]
  train_features = train_dataset[:, 0:FEATURE_NUMBER]

with open("../data/cancer_test.csv", "r") as f:
  test_dataset = np.loadtxt(f, delimiter=",")
  test_labels = test_dataset[:, FEATURE_NUMBER]
  test_features = test_dataset[:, 0:FEATURE_NUMBER]

# Define the model
classifier = tree.DecisionTreeClassifier()
#classifier = svm.SVC(C=1, kernel='linear')
#classifier = MLPClassifier(algorithm='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, learning_rate_init=0.001, batch_size=64, max_iter=100, verbose=False)
#classifier = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#classifier = MLPClassifier(algorithm='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# Train the model
print("Start to train")
model = classifier.fit(train_features, train_labels)

print("Start to validate")
predict_labels = model.predict(test_features)
auc = metrics.roc_auc_score(test_labels, predict_labels)
accuracy = metrics.accuracy_score(test_labels, predict_labels)

# Print the metrics
print("Accuracy: {}, acu: {}".format(accuracy, auc))
