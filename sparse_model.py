from __future__ import absolute_import, division, print_function

import tensorflow as tf


def full_connect(inputs,
                 weights_shape,
                 biases_shape,
                 is_train=True,
                 FLAGS=None):
  with tf.device("/cpu:0"):
    weights = tf.get_variable(
        "weights", weights_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable(
        "biases", biases_shape, initializer=tf.random_normal_initializer())
    layer = tf.matmul(inputs, weights) + biases

    if FLAGS.enable_bn and is_train:
      mean, var = tf.nn.moments(layer, axes=[0])
      scale = tf.get_variable(
          "scale", biases_shape, initializer=tf.random_normal_initializer())
      shift = tf.get_variable(
          "shift", biases_shape, initializer=tf.random_normal_initializer())
      layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                        FLAGS.bn_epsilon)
  return layer


def sparse_full_connect(sparse_ids,
                        sparse_values,
                        weights_shape,
                        biases_shape,
                        is_train=True,
                        FLAGS=None):

  weights = tf.get_variable(
      "weights", weights_shape, initializer=tf.random_normal_initializer())
  biases = tf.get_variable(
      "biases", biases_shape, initializer=tf.random_normal_initializer())
  return tf.nn.embedding_lookup_sparse(
      weights, sparse_ids, sparse_values, combiner="sum") + biases


def full_connect_relu(inputs,
                      weights_shape,
                      biases_shape,
                      is_train=True,
                      FLAGS=None):
  return tf.nn.relu(
      full_connect(inputs, weights_shape, biases_shape, is_train, FLAGS))


def customized_inference(sparse_ids, sparse_values, is_train=True, FLAGS=None):
  hidden1_units = 128
  hidden2_units = 32
  hidden3_units = 8

  with tf.variable_scope("input"):
    sparse_layer = sparse_full_connect(sparse_ids, sparse_values,
                                       [FLAGS.feature_size, hidden1_units],
                                       [hidden1_units], is_train, FLAGS)
    layer = tf.nn.relu(sparse_layer)
  with tf.variable_scope("layer0"):
    layer = full_connect_relu(layer, [hidden1_units, hidden2_units],
                              [hidden2_units], is_train, FLAGS)
  with tf.variable_scope("layer1"):
    layer = full_connect_relu(layer, [hidden2_units, hidden3_units],
                              [hidden3_units], is_train, FLAGS)
  if FLAGS.enable_dropout and is_train:
    layer = tf.nn.dropout(layer, FLAGS.dropout_keep_prob)
  with tf.variable_scope("output"):
    layer = full_connect(layer, [hidden3_units, FLAGS.label_size],
                         [FLAGS.label_size], is_train, FLAGS)
  return layer


def dnn_inference(sparse_ids, sparse_values, is_train=True, FLAGS=None):
  model_network_hidden_units = [int(i) for i in FLAGS.model_network.split()]

  with tf.variable_scope("input"):
    sparse_layer = sparse_full_connect(sparse_ids, sparse_values, [
        FLAGS.feature_size, model_network_hidden_units[0]
    ], [model_network_hidden_units[0]], is_train, FLAGS)
    layer = tf.nn.relu(sparse_layer)

  for i in range(len(model_network_hidden_units) - 1):
    with tf.variable_scope("layer{}".format(i)):
      layer = full_connect_relu(layer, [
          model_network_hidden_units[i], model_network_hidden_units[i + 1]
      ], [model_network_hidden_units[i + 1]], is_train, FLAGS)

  with tf.variable_scope("output"):
    layer = full_connect(layer,
                         [model_network_hidden_units[-1], FLAGS.label_size],
                         [FLAGS.label_size], is_train, FLAGS)
  return layer


def lr_inference(sparse_ids, sparse_values, is_train=True, FLAGS=None):
  with tf.variable_scope("logistic_regression"):
    layer = sparse_full_connect(sparse_ids, sparse_values,
                                [FLAGS.input_units, FLAGS.label_size],
                                [FLAGS.label_size], is_train, FLAGS)
  return layer


def wide_and_deep_inference(sparse_ids,
                            sparse_values,
                            is_train=True,
                            FLAGS=None):
  return lr_inference(sparse_ids,
                      sparse_values, is_train, FLAGS) + dnn_inference(
                          sparse_ids, sparse_values, is_train, FLAGS)
