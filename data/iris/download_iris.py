#!/usr/bin/env python

import numpy as np
import random
from sklearn import datasets


def main():
  # Load data set
  print("Download the iris dataset")
  iris = datasets.load_iris()

  # Split into train and test
  TRAIN_FILE_NAME = "iris_train.csv"
  TEST_FILE_NAME = "iris_test.csv"
  file_content_array = []
  train_file_content = ""
  test_file_content = ""
  total_number = len(iris.data)
  train_number = int(total_number * 0.9)
  test_number = total_number - train_number

  # Generate text content
  for i in range(total_number):
    line_content = str(iris.target[i]) + ","
    for j in iris.data[i]:
      line_content += str(j) + ","
    file_content_array.append(line_content[:-1] + "\n")

  random.shuffle(file_content_array)
  for i in file_content_array[:train_number]:
    train_file_content += i
  for i in file_content_array[train_number:]:
    test_file_content += i

  # Write into files
  print("Write content into files: {} and {}".format(TRAIN_FILE_NAME,
                                                     TEST_FILE_NAME))
  with open(TRAIN_FILE_NAME, "w") as f:
    f.write(train_file_content)
  with open(TEST_FILE_NAME, "w") as f:
    f.write(test_file_content)


if __name__ == "__main__":
  main()
