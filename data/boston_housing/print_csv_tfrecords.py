#!/usr/bin/env python

import tensorflow as tf
import os


def print_tfrecords(input_filename):
  max_print_number = 100
  current_print_number = 0

  for serialized_example in tf.python_io.tf_record_iterator(input_filename):
    # Get serialized example from file
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    label = example.features.feature["label"].float_list.value
    features = example.features.feature["features"].float_list.value
    print("Number: {}, label: {}, features: {}".format(current_print_number,
                                                       label, features))

    # Return when reaching max print number
    current_print_number += 1
    if current_print_number > max_print_number:
      exit()


def main():
  current_path = os.getcwd()
  tfrecords_file_name = "train.csv.tfrecords"
  input_filename = os.path.join(current_path, tfrecords_file_name)
  print_tfrecords(input_filename)


if __name__ == "__main__":
  main()
