#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent import futures
import time

import grpc

import inference_pb2

import tensorflow as tf

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class InferenceService(inference_pb2.InferenceServiceServicer):

  def __init__(self):

    input_units = 9
    hidden1_units = 10
    hidden2_units = 20
    output_units = 2

    weights1 = tf.Variable(
      tf.truncated_normal([input_units, hidden1_units]),
      dtype=tf.float32,
      name='weights')
    biases1 = tf.Variable(
      tf.truncated_normal([hidden1_units]),
      name='biases',
      dtype=tf.float32)

    # Hidden 2
    weights2 = tf.Variable(
      tf.truncated_normal([hidden1_units, hidden2_units]),
      dtype=tf.float32,
      name='weights')
    biases2 = tf.Variable(
      tf.truncated_normal([hidden2_units]),
      name='biases',
      dtype=tf.float32)

    # Linear
    weights3 = tf.Variable(
      tf.truncated_normal([hidden2_units, output_units]),
      dtype=tf.float32,
      name='weights')
    biases3 = tf.Variable(
      tf.truncated_normal([output_units]),
      name='biases',
      dtype=tf.float32)

    inference_features = tf.placeholder("float", [None, 9])
    inference_hidden1 = tf.nn.relu(tf.matmul(inference_features, weights1) +
                               biases1)
    inference_hidden2 = tf.nn.relu(tf.matmul(inference_hidden1, weights2) +
                               biases2)
    inference_logits = tf.matmul(inference_hidden2, weights3) + biases3
    inference_softmax = tf.nn.softmax(inference_logits)
    inference_op = tf.argmax(inference_softmax, 1)

    import numpy as np
    inference_data = np.array(
            [(10, 10, 10, 8, 6, 1, 8, 9, 1), (6, 2, 1, 1, 1, 1, 7, 1, 1),
             (2, 5, 3, 3, 6, 7, 7, 5, 1), (10, 4, 3, 1, 3, 3, 6, 5, 2),
             (6, 10, 10, 2, 8, 10, 7, 3, 3), (5, 6, 5, 6, 10, 1, 3, 1, 1),
             (1, 1, 1, 1, 2, 1, 2, 1, 2), (3, 7, 7, 4, 4, 9, 4, 8, 1),
             (1, 1, 1, 1, 2, 1, 2, 1, 1), (4, 1, 1, 3, 2, 1, 3, 1, 1)])
    correct_labels = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0]


    saver = tf.train.Saver()
    with tf.Session() as sess:
      # Restore wights from model file
      ckpt = tf.train.get_checkpoint_state("../checkpoint/")
      if ckpt and ckpt.model_checkpoint_path:
        print("Use the model {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        inference_result = sess.run(
                inference_op,
                feed_dict={inference_features: inference_data})
        print("Real data: {}".format(correct_labels))
        print("Inference data: {}".format(inference_result))
      else:
        print("No model found, exit")
        exit()

  def DoInference(self, request, context):
    return inference_pb2.InferenceResponse(message='Hello, %s!' % request.name)

def serve():
  #server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  #inference_pb2.add_InferenceServiceService_to_server(InferenceService(), server)
  server = inference_pb2.beta_create_InferenceService_server(InferenceService())
  server.add_insecure_port('[::]:50051')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  serve()
