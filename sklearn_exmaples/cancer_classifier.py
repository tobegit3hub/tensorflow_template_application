#!/usr/bin/env python

import sys
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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
classifiers = [
  DecisionTreeClassifier(max_depth=5),
  MLPClassifier(algorithm='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, learning_rate_init=0.001, batch_size=64, max_iter=100, verbose=False),
  MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
  MLPClassifier(algorithm='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
  KNeighborsClassifier(2),
  SVC(kernel="linear", C=0.025),
  SVC(gamma=2, C=1),
  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
  AdaBoostClassifier(),
  GaussianNB(),
  LinearDiscriminantAnalysis(),
  QuadraticDiscriminantAnalysis()]

if len(sys.argv) > 1:
  classifier_index = int(sys.argv[1])
else:
  classifier_index = 0
classifier = classifiers[classifier_index]
print("Use the classifier: {}".format(classifier))

# Train the model
print("Start to train")
model = classifier.fit(train_features, train_labels)

print("Start to validate")
predict_labels = model.predict(test_features)
auc = metrics.roc_auc_score(test_labels, predict_labels)
accuracy = metrics.accuracy_score(test_labels, predict_labels)

# Print the metrics
print("Accuracy: {}, acu: {}".format(accuracy, auc))
