#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


def generate_tfrecords_file(input_filename, output_filename):
  print("Start to convert {} to {}".format(input_filename, output_filename))

  writer = tf.python_io.TFRecordWriter(output_filename)

  for line in open(input_filename, "r"):
    data = line.split(",")
    label = int(data[-1])
    features = [float(i) for i in data[:-1]]

    example = tf.train.Example(features=tf.train.Features(
        feature={
            "label":
            tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "features":
            tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        }))
    writer.write(example.SerializeToString())

  writer.close()

  print(
      "Successfully convert {} to {}".format(input_filename, output_filename))


def main():
  current_path = os.getcwd()
  for filename in os.listdir(current_path):
    if filename.startswith("") and filename.endswith(".csv"):
      generate_tfrecords_file(filename, filename + ".tfrecords")


if __name__ == "__main__":
  main()
