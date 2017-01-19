#!/usr/bin/env python

import tensorflow as tf
import os


def generate_tfrecords(input_filename, output_filename):
  print("Start to convert {} to {}".format(input_filename, output_filename))
  writer = tf.python_io.TFRecordWriter(output_filename)

  for line in open(input_filename, "r"):
    data = line.split(",")
    label = float(data[9])
    features = [float(i) for i in data[:9]]

    example = tf.train.Example(features=tf.train.Features(feature={
        "label":
        tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        "features":
        tf.train.Feature(float_list=tf.train.FloatList(value=features)),
    }))
    writer.write(example.SerializeToString())

  writer.close()
  print("Successfully convert {} to {}".format(input_filename,
                                               output_filename))


def main():
  current_path = os.getcwd()
  for filename in os.listdir(current_path):
    if filename.startswith("") and filename.endswith(".csv"):
      generate_tfrecords(filename, filename + ".tfrecords")


if __name__ == "__main__":
  main()
