#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


def print_tfrecords_file(input_filename):
  print("Try to print the tfrecords file: {}".format(input_filename))

  max_print_number = 10
  current_print_index = 0

  for serialized_example in tf.python_io.tf_record_iterator(input_filename):
    # Get serialized example from file
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    label = example.features.feature["label"].int64_list.value
    ids = example.features.feature["ids"].int64_list.value
    values = example.features.feature["values"].float_list.value
    print("Index: {}, label: {}, features: {}".format(
            current_print_index, label, " ".join(
            [str(id) + ":" + str(value) for id, value in zip(ids, values)])))

    # Return when reaching max print number
    current_print_index += 1
    if current_print_index > max_print_number - 1:
      return


def main():
  current_path = os.getcwd()
  for filename in os.listdir(current_path):
    if filename.startswith("") and filename.endswith(".tfrecords"):
      tfrecords_file_path = os.path.join(current_path, filename)
      print_tfrecords_file(tfrecords_file_path)


if __name__ == "__main__":
  main()
