#!/usr/bin/env python

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 10, "The batch size to train")
flags.DEFINE_integer("epoch_number", 10, "Number of epochs to run trainer")
flags.DEFINE_integer("steps_to_validate", 1,
                     "Steps to validate and print loss")
flags.DEFINE_string("model_path", "./model/", "The export path of the model")
flags.DEFINE_integer("export_version", 1, "The version number of the model")

def main():
  # Define training data
  x = np.ones(FLAGS.batch_size)
  y = np.ones(FLAGS.batch_size)

  # Define the model
  X = tf.placeholder(tf.float32, shape=[None])
  Y = tf.placeholder(tf.float32, shape=[None])
  w = tf.Variable(1.0, name="weight")
  b = tf.Variable(1.0, name="bias")
  loss = tf.square(Y - tf.mul(X, w) - b)
  train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  predict_op  = tf.mul(X, w) + b

  saver = tf.train.Saver()

  # Start the session
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # Start training
    start_time = datetime.datetime.now()
    for epoch in range(FLAGS.epoch_number):
      sess.run(train_op, feed_dict={X: x, Y: y})

      # Start validating
      if epoch % FLAGS.steps_to_validate == 0:
        end_time = datetime.datetime.now()
        print("[{}] Epoch: {}".format(end_time - start_time, epoch))
        start_time = end_time

    # Print model variables
    w_value, b_value = sess.run([w, b])
    print("The model of w: {}, b: {}".format(w_value, b_value))

    # Export the model
    print("Exporting trained model to {}".format(FLAGS.model_path))
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
      sess.graph.as_graph_def(),
      named_graph_signatures={
        'inputs': exporter.generic_signature({"features": X}),
        'outputs': exporter.generic_signature({"prediction": predict_op})
      })
    model_exporter.export(FLAGS.model_path, tf.constant(FLAGS.export_version), sess)
    print 'Done exporting!'

if __name__ == "__main__":
  main()
