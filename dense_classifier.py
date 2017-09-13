#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import datetime
import json
import logging
import math
import numpy as np
import os
import pprint
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

# Define hyperparameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("enable_colored_log", False, "Enable colored log")
flags.DEFINE_string("input_file_format", "tfrecord", "Input file format")
flags.DEFINE_string("train_file", "./data/cancer/cancer_train.csv.tfrecords",
                    "The glob pattern of train TFRecords files")
flags.DEFINE_string("validate_file", "./data/cancer/cancer_test.csv.tfrecords",
                    "The glob pattern of validate TFRecords files")
flags.DEFINE_integer("feature_size", 9, "Number of feature size")
flags.DEFINE_integer("label_size", 2, "Number of label size")
flags.DEFINE_float("learning_rate", 0.01, "The learning rate")
flags.DEFINE_integer("epoch_number", 1000, "Number of epochs to train")
flags.DEFINE_integer("batch_size", 1024, "The batch size of training")
flags.DEFINE_integer("validate_batch_size", 1024,
                     "The batch size of validation")
flags.DEFINE_integer("batch_thread_number", 1,
                     "Number of threads to read data")
flags.DEFINE_integer("min_after_dequeue", 100,
                     "The minimal number after dequeue")
flags.DEFINE_string("checkpoint_path", "./checkpoint/",
                    "The path of checkpoint")
flags.DEFINE_string("output_path", "./tensorboard/",
                    "The path of tensorboard event files")
flags.DEFINE_string("scenario", "classification",
                    "Support classification and regression")
flags.DEFINE_string("model", "dnn", "Support dnn, lr, wide_and_deep")
flags.DEFINE_string("model_network", "128 32 8", "The neural network of model")
flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization or not")
flags.DEFINE_float("bn_epsilon", 0.001, "The epsilon of batch normalization")
flags.DEFINE_boolean("enable_dropout", False, "Enable dropout or not")
flags.DEFINE_float("dropout_keep_prob", 0.5, "The dropout keep prob")
flags.DEFINE_boolean("enable_lr_decay", False, "Enable learning rate decay")
flags.DEFINE_float("lr_decay_rate", 0.96, "Learning rate decay rate")
flags.DEFINE_string("optimizer", "adagrad", "The optimizer to train")
flags.DEFINE_integer("steps_to_validate", 10,
                     "Steps to validate and print state")
flags.DEFINE_string("mode", "train",
                    "Support train, export, inference, savedmodel")
flags.DEFINE_string("model_path", "./model/", "The path of the model")
flags.DEFINE_string("saved_model_path", "./saved_model/",
                    "The path of the saved model")
flags.DEFINE_integer("model_version", 1, "The version of the model")
flags.DEFINE_string("inference_test_file", "./data/cancer_test.csv",
                    "The test file for inference")
flags.DEFINE_string("inference_result_file", "./inference_result.txt",
                    "The result file from inference")
flags.DEFINE_boolean("benchmark_mode", False,
                     "Reduce extra computation in benchmark mode")


def main():
  # Get hyperparameters
  if FLAGS.enable_colored_log:
    import coloredlogs
    coloredlogs.install()
  logging.basicConfig(level=logging.INFO)
  INPUT_FILE_FORMAT = FLAGS.input_file_format
  if INPUT_FILE_FORMAT not in ["tfrecord", "csv"]:
    logging.error("Unknow input file format: {}".format(INPUT_FILE_FORMAT))
    exit(1)
  FEATURE_SIZE = FLAGS.feature_size
  LABEL_SIZE = FLAGS.label_size
  EPOCH_NUMBER = FLAGS.epoch_number
  if EPOCH_NUMBER <= 0:
    EPOCH_NUMBER = None
  BATCH_THREAD_NUMBER = FLAGS.batch_thread_number
  MIN_AFTER_DEQUEUE = FLAGS.min_after_dequeue
  BATCH_CAPACITY = BATCH_THREAD_NUMBER * FLAGS.batch_size + MIN_AFTER_DEQUEUE
  MODE = FLAGS.mode
  SCENARIO = FLAGS.scenario
  MODEL = FLAGS.model
  CHECKPOINT_PATH = FLAGS.checkpoint_path
  if not CHECKPOINT_PATH.startswith("fds://") and not os.path.exists(
      CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
  CHECKPOINT_FILE = CHECKPOINT_PATH + "/checkpoint.ckpt"
  LATEST_CHECKPOINT = tf.train.latest_checkpoint(CHECKPOINT_PATH)
  OUTPUT_PATH = FLAGS.output_path
  if not OUTPUT_PATH.startswith("fds://") and not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
  pprint.PrettyPrinter().pprint(FLAGS.__flags)

  # Process TFRecoreds files
  def read_and_decode_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.float32),
            "features": tf.FixedLenFeature([FEATURE_SIZE], tf.float32),
        })
    label = features["label"]
    features = features["features"]
    return label, features

  def read_and_decode_csv(filename_queue):
    # TODO: Not generic for all datasets
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    #record_defaults = [[1], [1], [1], [1], [1]]
    record_defaults = [[1], [1.0], [1.0], [1.0], [1.0]]
    col1, col2, col3, col4, col5 = tf.decode_csv(
        value, record_defaults=record_defaults)
    label = col1
    features = tf.stack([col2, col3, col4, col4])
    return label, features

  # Read TFRecords files for training
  filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(FLAGS.train_file), num_epochs=EPOCH_NUMBER)
  if INPUT_FILE_FORMAT == "tfrecord":
    label, features = read_and_decode_tfrecord(filename_queue)
  elif INPUT_FILE_FORMAT == "csv":
    label, features = read_and_decode_csv(filename_queue)
  batch_labels, batch_features = tf.train.shuffle_batch(
      [label, features],
      batch_size=FLAGS.batch_size,
      num_threads=BATCH_THREAD_NUMBER,
      capacity=BATCH_CAPACITY,
      min_after_dequeue=MIN_AFTER_DEQUEUE)

  # Read TFRecords file for validatioin
  validate_filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(FLAGS.validate_file),
      num_epochs=EPOCH_NUMBER)
  if INPUT_FILE_FORMAT == "tfrecord":
    validate_label, validate_features = read_and_decode_tfrecord(
        validate_filename_queue)
  elif INPUT_FILE_FORMAT == "csv":
    validate_label, validate_features = read_and_decode_csv(
        validate_filename_queue)
  validate_batch_labels, validate_batch_features = tf.train.shuffle_batch(
      [validate_label, validate_features],
      batch_size=FLAGS.validate_batch_size,
      num_threads=BATCH_THREAD_NUMBER,
      capacity=BATCH_CAPACITY,
      min_after_dequeue=MIN_AFTER_DEQUEUE)

  # Define the model
  input_units = FEATURE_SIZE
  output_units = LABEL_SIZE
  model_network_hidden_units = [int(i) for i in FLAGS.model_network.split()]

  def full_connect(inputs, weights_shape, biases_shape, is_train=True):
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

  def full_connect_relu(inputs, weights_shape, biases_shape, is_train=True):
    layer = full_connect(inputs, weights_shape, biases_shape, is_train)
    layer = tf.nn.relu(layer)
    return layer

  def customized_inference(inputs, is_train=True):
    hidden1_units = 128
    hidden2_units = 32
    hidden3_units = 8

    with tf.variable_scope("input"):
      layer = full_connect_relu(inputs, [input_units, hidden1_units],
                                [hidden1_units], is_train)
    with tf.variable_scope("layer0"):
      layer = full_connect_relu(layer, [hidden1_units, hidden2_units],
                                [hidden2_units], is_train)
    with tf.variable_scope("layer1"):
      layer = full_connect_relu(layer, [hidden2_units, hidden3_units],
                                [hidden3_units], is_train)
    if FLAGS.enable_dropout and is_train:
      layer = tf.nn.dropout(layer, FLAGS.dropout_keep_prob)
    with tf.variable_scope("output"):
      layer = full_connect(layer, [hidden3_units, output_units],
                           [output_units], is_train)
    return layer

  def dnn_inference(inputs, is_train=True):
    with tf.variable_scope("input"):
      layer = full_connect_relu(inputs,
                                [input_units, model_network_hidden_units[0]],
                                [model_network_hidden_units[0]], is_train)

    for i in range(len(model_network_hidden_units) - 1):
      with tf.variable_scope("layer{}".format(i)):
        layer = full_connect_relu(layer, [
            model_network_hidden_units[i], model_network_hidden_units[i + 1]
        ], [model_network_hidden_units[i + 1]], is_train)

    with tf.variable_scope("output"):
      layer = full_connect(layer,
                           [model_network_hidden_units[-1],
                            output_units], [output_units], is_train)
    return layer

  def lr_inference(inputs, is_train=True):
    with tf.variable_scope("lr"):
      layer = full_connect(inputs, [input_units, output_units], [output_units])
    return layer

  def wide_and_deep_inference(inputs, is_train=True):
    return lr_inference(inputs, is_train) + dnn_inference(inputs, is_train)

  def cnn_inference(inputs, is_train=True):
    # TODO: Change if validate_batch_size is different
    # [BATCH_SIZE, 512 * 512 * 1] -> [BATCH_SIZE, 512, 512, 1]
    inputs = tf.reshape(inputs, [FLAGS.batch_size, 512, 512, 1])

    # [BATCH_SIZE, 512, 512, 1] -> [BATCH_SIZE, 128, 128, 8]
    with tf.variable_scope("conv0"):
      weights = tf.get_variable(
          "weights", [3, 3, 1, 8], initializer=tf.random_normal_initializer())
      bias = tf.get_variable(
          "bias", [8], initializer=tf.random_normal_initializer())

      layer = tf.nn.conv2d(
          inputs, weights, strides=[1, 1, 1, 1], padding="SAME")
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

      layer = tf.nn.conv2d(
          layer, weights, strides=[1, 1, 1, 1], padding="SAME")
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

      layer = tf.nn.conv2d(
          layer, weights, strides=[1, 1, 1, 1], padding="SAME")
      layer = tf.nn.bias_add(layer, bias)
      layer = tf.nn.relu(layer)
      layer = tf.nn.max_pool(
          layer, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    # [BATCH_SIZE, 8, 8, 8] -> [BATCH_SIZE, 8 * 8 * 8]
    layer = tf.reshape(layer, [-1, 8 * 8 * 8])

    # [BATCH_SIZE, 8 * 8 * 8] -> [BATCH_SIZE, LABEL_SIZE]
    with tf.variable_scope("output"):
      weights = tf.get_variable(
          "weights", [8 * 8 * 8, LABEL_SIZE],
          initializer=tf.random_normal_initializer())
      bias = tf.get_variable(
          "bias", [LABEL_SIZE], initializer=tf.random_normal_initializer())
      layer = tf.add(tf.matmul(layer, weights), bias)

    return layer

  def inference(inputs, is_train=True):
    if MODEL == "dnn":
      return dnn_inference(inputs, is_train)
    elif MODEL == "lr":
      return lr_inference(inputs, is_train)
    elif MODEL == "wide_and_deep":
      return wide_and_deep_inference(inputs, is_train)
    elif MODEL == "customized":
      return customized_inference(inputs, is_train)
    elif MODEL == "cnn":
      return cnn_inference(inputs, is_train)
    else:
      logging.error("Unknown model, exit now")
      exit(1)

  logging.info("Use the model: {}, model network: {}".format(
      MODEL, FLAGS.model_network))
  logits = inference(batch_features, True)

  if SCENARIO == "classification":
    batch_labels = tf.to_int64(batch_labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=batch_labels)
    loss = tf.reduce_mean(cross_entropy, name="loss")
  elif SCENARIO == "regression":
    msl = tf.square(logits - batch_labels, name="msl")
    loss = tf.reduce_mean(msl, name="loss")
  else:
    logging.error("Unknow scenario: {}".format(SCENARIO))
    return

  global_step = tf.Variable(0, name="global_step", trainable=False)
  if FLAGS.enable_lr_decay:
    logging.info(
        "Enable learning rate decay rate: {}".format(FLAGS.lr_decay_rate))
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(
        starter_learning_rate,
        global_step,
        100000,
        FLAGS.lr_decay_rate,
        staircase=True)
  else:
    learning_rate = FLAGS.learning_rate
  optimizer = get_optimizer(FLAGS.optimizer, learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)
  tf.get_variable_scope().reuse_variables()

  # Avoid error when not using acc and auc op
  if SCENARIO == "regression":
    batch_labels = tf.to_int64(batch_labels)

  # Define accuracy op for train data
  train_accuracy_logits = inference(batch_features, False)
  train_softmax = tf.nn.softmax(train_accuracy_logits)
  train_correct_prediction = tf.equal(
      tf.argmax(train_softmax, 1), batch_labels)
  train_accuracy = tf.reduce_mean(
      tf.cast(train_correct_prediction, tf.float32))

  # Define auc op for train data
  batch_labels = tf.cast(batch_labels, tf.int32)
  sparse_labels = tf.reshape(batch_labels, [-1, 1])
  derived_size = tf.shape(batch_labels)[0]
  indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
  concated = tf.concat(axis=1, values=[indices, sparse_labels])
  outshape = tf.stack([derived_size, LABEL_SIZE])
  new_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
  _, train_auc = tf.contrib.metrics.streaming_auc(train_softmax,
                                                  new_batch_labels)

  # Define accuracy op for validate data
  validate_accuracy_logits = inference(validate_batch_features, False)
  validate_softmax = tf.nn.softmax(validate_accuracy_logits)
  validate_batch_labels = tf.to_int64(validate_batch_labels)
  validate_correct_prediction = tf.equal(
      tf.argmax(validate_softmax, 1), validate_batch_labels)
  validate_accuracy = tf.reduce_mean(
      tf.cast(validate_correct_prediction, tf.float32))

  # Define auc op for validate data
  validate_batch_labels = tf.cast(validate_batch_labels, tf.int32)
  sparse_labels = tf.reshape(validate_batch_labels, [-1, 1])
  derived_size = tf.shape(validate_batch_labels)[0]
  indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
  concated = tf.concat(axis=1, values=[indices, sparse_labels])
  outshape = tf.stack([derived_size, LABEL_SIZE])
  new_validate_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
  _, validate_auc = tf.contrib.metrics.streaming_auc(validate_softmax,
                                                     new_validate_batch_labels)

  # Define inference op
  inference_features = tf.placeholder("float", [None, FEATURE_SIZE])
  inference_logits = inference(inference_features, False)
  inference_softmax = tf.nn.softmax(inference_logits)
  inference_op = tf.argmax(inference_softmax, 1)
  keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
  keys = tf.identity(keys_placeholder)
  model_signature = {
      "inputs":
      exporter.generic_signature({
          "keys": keys_placeholder,
          "features": inference_features
      }),
      "outputs":
      exporter.generic_signature({
          "keys": keys,
          "softmax": inference_softmax,
          "prediction": inference_op
      })
  }

  # Initialize saver and summary
  saver = tf.train.Saver()
  tf.summary.scalar("loss", loss)
  if SCENARIO == "classification":
    tf.summary.scalar("train_accuracy", train_accuracy)
    tf.summary.scalar("train_auc", train_auc)
    tf.summary.scalar("validate_accuracy", validate_accuracy)
    tf.summary.scalar("validate_auc", validate_auc)
  summary_op = tf.summary.merge_all()
  init_op = [
      tf.global_variables_initializer(),
      tf.local_variables_initializer()
  ]

  # Create session to run
  with tf.Session() as sess:
    logging.info("Start to run with mode: {}".format(MODE))
    writer = tf.summary.FileWriter(OUTPUT_PATH, sess.graph)
    sess.run(init_op)

    if MODE == "train":
      # Restore session and start queue runner
      restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord, sess=sess)
      start_time = datetime.datetime.now()

      try:
        while not coord.should_stop():
          if FLAGS.benchmark_mode:
            sess.run(train_op)
          else:
            _, step = sess.run([train_op, global_step])

            # Print state while training
            if step % FLAGS.steps_to_validate == 0:
              if SCENARIO == "classification":
                loss_value, train_accuracy_value, train_auc_value, validate_accuracy_value, validate_auc_value, summary_value = sess.run(
                    [
                        loss, train_accuracy, train_auc, validate_accuracy,
                        validate_auc, summary_op
                    ])
                end_time = datetime.datetime.now()
                logging.info(
                    "[{}] Step: {}, loss: {}, train_acc: {}, train_auc: {}, valid_acc: {}, valid_auc: {}".
                    format(end_time - start_time, step, loss_value,
                           train_accuracy_value, train_auc_value,
                           validate_accuracy_value, validate_auc_value))
              elif SCENARIO == "regression":
                loss_value, summary_value = sess.run([loss, summary_op])
                end_time = datetime.datetime.now()
                logging.info("[{}] Step: {}, loss: {}".format(
                    end_time - start_time, step, loss_value))

              writer.add_summary(summary_value, step)
              saver.save(sess, CHECKPOINT_FILE, global_step=step)
              start_time = end_time
      except tf.errors.OutOfRangeError:
        if FLAGS.benchmark_mode:
          print("Finish training for benchmark")
          exit(0)
        else:
          # Export the model after training
          export_model(sess, saver, model_signature, FLAGS.model_path,
                       FLAGS.model_version)
      finally:
        coord.request_stop()
      coord.join(threads)

    elif MODE == "export":
      if not restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT):
        logging.error("No checkpoint found, exit now")
        exit(1)

      # Export the model
      export_model(sess, saver, model_signature, FLAGS.model_path,
                   FLAGS.model_version)

    elif MODE == "savedmodel":
      if not restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT):
        logging.error("No checkpoint found, exit now")
        exit(1)

      logging.info(
          "Export the saved model to {}".format(FLAGS.saved_model_path))
      export_path_base = FLAGS.saved_model_path
      export_path = os.path.join(
          compat.as_bytes(export_path_base),
          compat.as_bytes(str(FLAGS.model_version)))

      model_signature = signature_def_utils.build_signature_def(
          inputs={
              "keys": utils.build_tensor_info(keys_placeholder),
              "features": utils.build_tensor_info(inference_features)
          },
          outputs={
              "keys": utils.build_tensor_info(keys),
              "softmax": utils.build_tensor_info(inference_softmax),
              "prediction": utils.build_tensor_info(inference_op)
          },
          method_name=signature_constants.PREDICT_METHOD_NAME)

      try:
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess,
            [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                model_signature,
            },
            #legacy_init_op=legacy_init_op)
            legacy_init_op=tf.group(
                tf.initialize_all_tables(), name="legacy_init_op"))

        builder.save()
      except Exception as e:
        logging.error("Fail to export saved model, exception: {}".format(e))

    elif MODE == "inference":
      if not restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT):
        logging.error("No checkpoint found, exit now")
        exit(1)

      # Load inference test data
      inference_result_file_name = FLAGS.inference_result_file
      inference_test_file_name = FLAGS.inference_test_file
      inference_data = np.genfromtxt(inference_test_file_name, delimiter=",")
      inference_data_features = inference_data[:, 0:9]
      inference_data_labels = inference_data[:, 9]

      # Run inference
      start_time = datetime.datetime.now()
      prediction, prediction_softmax = sess.run(
          [inference_op, inference_softmax],
          feed_dict={inference_features: inference_data_features})
      end_time = datetime.datetime.now()

      # Compute accuracy
      label_number = len(inference_data_labels)
      correct_label_number = 0
      for i in range(label_number):
        if inference_data_labels[i] == prediction[i]:
          correct_label_number += 1
      accuracy = float(correct_label_number) / label_number

      # Compute auc
      y_true = np.array(inference_data_labels)
      y_score = prediction_softmax[:, 1]
      fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
      auc = metrics.auc(fpr, tpr)
      logging.info("[{}] Inference accuracy: {}, auc: {}".format(
          end_time - start_time, accuracy, auc))

      # Save result into the file
      np.savetxt(inference_result_file_name, prediction_softmax, delimiter=",")
      logging.info(
          "Save result to file: {}".format(inference_result_file_name))


def get_optimizer(optimizer, learning_rate):
  logging.info("Use the optimizer: {}".format(optimizer))
  if optimizer == "sgd":
    return tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer == "adadelta":
    return tf.train.AdadeltaOptimizer(learning_rate)
  elif optimizer == "adagrad":
    return tf.train.AdagradOptimizer(learning_rate)
  elif optimizer == "adam":
    return tf.train.AdamOptimizer(learning_rate)
  elif optimizer == "ftrl":
    return tf.train.FtrlOptimizer(learning_rate)
  elif optimizer == "rmsprop":
    return tf.train.RMSPropOptimizer(learning_rate)
  else:
    logging.error("Unknow optimizer, exit now")
    exit(1)


def restore_session_from_checkpoint(sess, saver, checkpoint):
  if checkpoint:
    logging.info("Restore session from checkpoint: {}".format(checkpoint))
    saver.restore(sess, checkpoint)
    return True
  else:
    return False


def export_model(sess, saver, signature, model_path, model_version):
  logging.info("Export the model to {}".format(model_path))
  model_exporter = exporter.Exporter(saver)
  model_exporter.init(
      sess.graph.as_graph_def(),
      named_graph_signatures=signature,
      clear_devices=True)
  try:
    model_exporter.export(model_path, tf.constant(model_version), sess)
  except Exception as e:
    logging.error("Fail to export model, exception: {}".format(e))


if __name__ == "__main__":
  main()
