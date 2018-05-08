from __future__ import absolute_import, division, print_function

import tensorflow as tf


def full_connect(inputs,
                 weights_shape,
                 biases_shape,
                 is_train=True,
                 FLAGS=None):
  """
    Define full-connect layer with reused Variables.
    """

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


def full_connect_relu(inputs,
                      weights_shape,
                      biases_shape,
                      is_train=True,
                      FLAGS=None):
  """
    Define full-connect layer and activation function with reused Variables.
    """

  layer = full_connect(inputs, weights_shape, biases_shape, is_train, FLAGS)
  layer = tf.nn.relu(layer)
  return layer


def customized_inference(inputs,
                         input_units,
                         output_units,
                         is_train=True,
                         FLAGS=None):
  """
    Define the customed model.
    """

  hidden1_units = 128
  hidden2_units = 32
  hidden3_units = 8

  with tf.variable_scope("input_layer"):
    layer = full_connect_relu(inputs, [input_units, hidden1_units],
                              [hidden1_units], is_train, FLAGS)
  with tf.variable_scope("layer_0"):
    layer = full_connect_relu(layer, [hidden1_units, hidden2_units],
                              [hidden2_units], is_train, FLAGS)
  with tf.variable_scope("layer_1"):
    layer = full_connect_relu(layer, [hidden2_units, hidden3_units],
                              [hidden3_units], is_train, FLAGS)
  if FLAGS.enable_dropout and is_train:
    layer = tf.nn.dropout(layer, FLAGS.dropout_keep_prob)
  with tf.variable_scope("output_layer"):
    layer = full_connect(layer, [hidden3_units, output_units], [output_units],
                         is_train, FLAGS)
  return layer


def dnn_inference(inputs, input_units, output_units, is_train=True,
                  FLAGS=None):
  """
    Define the DNN model.
    """

  # Example: [128, 64, 32, 16]
  model_network_hidden_units = [int(i) for i in FLAGS.dnn_struct.split()]
  with tf.variable_scope("input_layer"):
    layer = full_connect_relu(inputs,
                              [input_units, model_network_hidden_units[0]],
                              [model_network_hidden_units[0]], is_train, FLAGS)

  for i in range(len(model_network_hidden_units) - 1):
    with tf.variable_scope("layer_{}".format(i)):
      layer = full_connect_relu(layer, [
          model_network_hidden_units[i], model_network_hidden_units[i + 1]
      ], [model_network_hidden_units[i + 1]], is_train, FLAGS)

  with tf.variable_scope("output_layer"):
    layer = full_connect(layer, [model_network_hidden_units[-1], output_units],
                         [output_units], is_train, FLAGS)
  return layer


def lr_inference(inputs, input_units, output_units, is_train=True, FLAGS=None):
  """
    Define the linear regression model.
    """

  with tf.variable_scope("lr"):
    layer = full_connect(inputs, [input_units, output_units], [output_units],
                         FLAGS)
  return layer


def wide_and_deep_inference(inputs,
                            input_units,
                            output_units,
                            is_train=True,
                            FLAGS=None):
  """
    Define the wide-and-deep model.
    """

  return lr_inference(inputs, input_units,
                      output_units, is_train, FLAGS) + dnn_inference(
                          inputs, input_units, output_units, is_train, FLAGS)


def cnn_inference(inputs, input_units, output_units, is_train=True,
                  FLAGS=None):
  """
    Define the CNN model.
    """

  # TODO: Change if validate_batch_size is different
  # [BATCH_SIZE, 512 * 512 * 1] -> [BATCH_SIZE, 512, 512, 1]
  inputs = tf.reshape(inputs, [FLAGS.batch_size, 512, 512, 1])

  # [BATCH_SIZE, 512, 512, 1] -> [BATCH_SIZE, 128, 128, 8]
  with tf.variable_scope("conv0"):
    weights = tf.get_variable(
        "weights", [3, 3, 1, 8], initializer=tf.random_normal_initializer())
    bias = tf.get_variable(
        "bias", [8], initializer=tf.random_normal_initializer())

    layer = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding="SAME")
    layer = tf.nn.bias_add(layer, bias)
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(
        layer, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

  # [BATCH_SIZE, 128, 128, 8] -> [BATCH_SIZE, 32, 32, 8]
  with tf.variable_scope("conv1"):
    weights = tf.get_variable(
        "weights", [3, 3, 8, 8], initializer=tf.random_normal_initializer())
    bias = tf.get_variable(
        "bias", [8], initializer=tf.random_normal_initializer())

    layer = tf.nn.conv2d(layer, weights, strides=[1, 1, 1, 1], padding="SAME")
    layer = tf.nn.bias_add(layer, bias)
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(
        layer, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

  # [BATCH_SIZE, 32, 32, 8] -> [BATCH_SIZE, 8, 8, 8]
  with tf.variable_scope("conv2"):
    weights = tf.get_variable(
        "weights", [3, 3, 8, 8], initializer=tf.random_normal_initializer())
    bias = tf.get_variable(
        "bias", [8], initializer=tf.random_normal_initializer())

    layer = tf.nn.conv2d(layer, weights, strides=[1, 1, 1, 1], padding="SAME")
    layer = tf.nn.bias_add(layer, bias)
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(
        layer, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

  # [BATCH_SIZE, 8, 8, 8] -> [BATCH_SIZE, 8 * 8 * 8]
  layer = tf.reshape(layer, [-1, 8 * 8 * 8])

  # [BATCH_SIZE, 8 * 8 * 8] -> [BATCH_SIZE, LABEL_SIZE]
  with tf.variable_scope("output"):
    weights = tf.get_variable(
        "weights", [8 * 8 * 8, FLAGS.label_size],
        initializer=tf.random_normal_initializer())
    bias = tf.get_variable(
        "bias", [FLAGS.label_size], initializer=tf.random_normal_initializer())
    layer = tf.add(tf.matmul(layer, weights), bias)

  return layer
