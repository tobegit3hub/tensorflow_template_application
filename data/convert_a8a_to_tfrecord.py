#!/usr/bin/env python

import tensorflow as tf
import os

# The data in a8a_train.libsvm:
# 0 5:1 6:1 17:1 21:1 35:1 40:1 53:1 63:1 71:1 73:1 74:1 76:1 80:1 83:1
# 1 5:1 7:1 17:1 22:1 36:1 40:1 51:1 63:1 67:1 73:1 74:1 76:1 81:1 83:1
# 1 2:1 6:1 14:1 29:1 39:1 42:1 52:1 64:1 67:1 72:1 75:1 76:1 82:1 83:1
# 1 4:1 6:1 16:1 19:1 39:1 40:1 51:1 63:1 67:1 73:1 75:1 76:1 80:1 83:1


def convert_tfrecords(input_filename, output_filename):
    current_path = os.getcwd()
    input_file = os.path.join(current_path, input_filename)
    output_file = os.path.join(current_path, output_filename)
    print("Start to convert {} to {}".format(input_file, output_file))

    writer = tf.python_io.TFRecordWriter(output_file)

    for line in open(input_file, "r"):
        data = line.split(" ")
        label = float(data[0])
        ids = []
        values = []
        for fea in data[1:]:
            id, value = fea.split(":")
            ids.append(int(id))
            values.append(float(value))

        # Write each example one by one
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":
                tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            "ids":
                tf.train.Feature(int64_list=tf.train.Int64List(value=ids)),
            "values":
                tf.train.Feature(float_list=tf.train.FloatList(value=values))
        }))

        writer.write(example.SerializeToString())

    writer.close()
    print("Successfully convert {} to {}".format(input_file, output_file))


current_path = os.getcwd()
for file in os.listdir(current_path):
    if file.startswith("a8a") and file.endswith(".libsvm"):
        convert_tfrecords(file, file + ".tfrecords")