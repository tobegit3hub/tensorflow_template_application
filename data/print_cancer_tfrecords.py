#!/usr/bin/env python

import tensorflow as tf
import os

# Read TFRecords file
current_path = os.getcwd()
tfrecords_file_name = "cancer.csv.tfrecords"
#tfrecords_file_name = "cancer_test.csv.tfrecords"
input_file = os.path.join(current_path, tfrecords_file_name)

# Constrain the data to print
max_print_number = 100
print_number = 1

for serialized_example in tf.python_io.tf_record_iterator(input_file):
    # Get serialized example from file
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    # Read data in specified format
    label = example.features.feature["label"].float_list.value
    features = example.features.feature["features"].float_list.value
    print("Number: {}, label: {}, features: {}".format(print_number, label,
                                                       features))

    # Return when reaching max print number
    if print_number > max_print_number:
        exit()
    else:
        print_number += 1
