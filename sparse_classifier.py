#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import datetime
import logging
import os
import pprint
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)

import sparse_model
import util

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def define_flags():
  """
    Define all the command-line parameters.
    
    Return:
      The FLAGS object.
    """

  flags = tf.app.flags
  flags.DEFINE_string("train_files", "./data/a8a/a8a_train.libsvm.tfrecords",
                      "The glob pattern of train TFRecords files")
  flags.DEFINE_string("validation_files",
                      "./data/a8a/a8a_test.libsvm.tfrecords",
                      "The glob pattern of train TFRecords files")
  flags.DEFINE_integer("feature_size", 124, "Number of feature size")
  flags.DEFINE_integer("label_size", 2, "Number of label size")
  flags.DEFINE_string("label_type", "int", "The type of label")
  flags.DEFINE_float("learning_rate", 0.01, "The learning rate")
  flags.DEFINE_integer("epoch_number", 10, "Number of epochs to train")
  flags.DEFINE_integer("batch_size", 1024, "The batch size of training")
  flags.DEFINE_integer("train_batch_size", 64, "The batch size of training")
  flags.DEFINE_integer("validation_batch_size", 64,
                       "The batch size of training")
  flags.DEFINE_integer("validate_batch_size", 1024,
                       "The batch size of validation")
  flags.DEFINE_integer("batch_thread_number", 1,
                       "Number of threads to read data")
  flags.DEFINE_integer("min_after_dequeue", 100,
                       "The minimal number after dequeue")
  flags.DEFINE_string("checkpoint_path", "./sparse_checkpoint/",
                      "The path of checkpoint")
  flags.DEFINE_string("output_path", "./sparse_tensorboard/",
                      "The path of tensorboard event files")
  flags.DEFINE_string("model", "dnn", "Support dnn, lr, wide_and_deep")
  flags.DEFINE_string("model_network", "128 32 8",
                      "The neural network of model")
  flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization or not")
  flags.DEFINE_float("bn_epsilon", 0.001, "The epsilon of batch normalization")
  flags.DEFINE_boolean("enable_dropout", False, "Enable dropout or not")
  flags.DEFINE_float("dropout_keep_prob", 0.5, "The dropout keep prob")
  flags.DEFINE_boolean("enable_lr_decay", False, "Enable learning rate decay")
  flags.DEFINE_float("lr_decay_rate", 0.96, "Learning rate decay rate")
  flags.DEFINE_string("optimizer", "adagrad", "The optimizer to train")
  flags.DEFINE_integer("steps_to_validate", 10,
                       "Steps to validate and print state")
  flags.DEFINE_string("mode", "train", "Support train, export, inference")
  flags.DEFINE_string("saved_model_path", "./sparse_saved_model/",
                      "The path of the saved model")
  flags.DEFINE_string("model_path", "./sparse_model/", "The path of the model")
  flags.DEFINE_integer("model_version", 1, "The version of the model")
  flags.DEFINE_string("inference_test_file", "./data/a8a_test.libsvm",
                      "The test file for inference")
  flags.DEFINE_string("inference_result_file", "./inference_result.txt",
                      "The result file from inference")
  flags.DEFINE_boolean("benchmark_mode", False,
                       "Reduce extra computation in benchmark mode")
  FLAGS = flags.FLAGS

  # Check parameters
  assert (FLAGS.optimizer in [
      "sgd", "adadelta", "adagrad", "adam", "ftrl", "rmsprop"
  ])

  # Print flags
  FLAGS.mode
  parameter_value_map = {}
  for key in FLAGS.__flags.keys():
    parameter_value_map[key] = FLAGS.__flags[key].value
  pprint.PrettyPrinter().pprint(parameter_value_map)
  return FLAGS


FLAGS = define_flags()


def parse_tfrecords_function(example_proto):
  """
    Decode TFRecords for Dataset.
    
    Args:
      example_proto: TensorFlow ExampleProto object. 
    
    Return:
      The op of features and labels
    """

  if FLAGS.label_type == "int":
    features = {
        "ids": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32),
        "label": tf.FixedLenFeature([], tf.int64, default_value=0)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    labels = parsed_features["label"]
    ids = parsed_features["ids"]
    values = parsed_features["values"]

  elif FLAGS.label_type == "float":
    features = {
        "ids": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32),
        "label": tf.FixedLenFeature([], tf.float32, default_value=0)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    labels = tf.cast(parsed_features["label"], tf.int32)
    ids = parsed_features["ids"]
    values = parsed_features["values"]

  return labels, ids, values


def inference(sparse_ids, sparse_values, is_train=True):
  """
    Define the model by model name.
    
    Return:
      The logit of the model output.
    """

  if FLAGS.model == "dnn":
    return sparse_model.dnn_inference(sparse_ids, sparse_values, is_train,
                                      FLAGS)
  elif FLAGS.model == "lr":
    return sparse_model.lr_inference(sparse_ids, sparse_values, is_train,
                                     FLAGS)
  elif FLAGS.model == "wide_and_deep":
    return sparse_model.wide_and_deep_inference(sparse_ids, sparse_values,
                                                is_train, FLAGS)
  elif FLAGS.model == "customized":
    return sparse_model.customized_inference(sparse_ids, sparse_values,
                                             is_train, FLAGS)


def main():

  if os.path.exists(FLAGS.checkpoint_path) == False:
    os.makedirs(FLAGS.checkpoint_path)
  checkpoint_file_path = FLAGS.checkpoint_path + "/checkpoint.ckpt"
  latest_checkpoint_file_path = tf.train.latest_checkpoint(
      FLAGS.checkpoint_path)

  if os.path.exists(FLAGS.output_path) == False:
    os.makedirs(FLAGS.output_path)

  # Step 1: Construct the dataset op
  epoch_number = FLAGS.epoch_number
  if epoch_number <= 0:
    epoch_number = -1
  train_buffer_size = FLAGS.train_batch_size * 3
  validation_buffer_size = FLAGS.train_batch_size * 3

  train_filename_list = [filename for filename in FLAGS.train_files.split(",")]
  train_filename_placeholder = tf.placeholder(tf.string, shape=[None])
  train_dataset = tf.data.TFRecordDataset(train_filename_placeholder)
  train_dataset = train_dataset.map(parse_tfrecords_function).repeat(
      epoch_number).batch(FLAGS.train_batch_size).shuffle(
          buffer_size=train_buffer_size)
  train_dataset_iterator = train_dataset.make_initializable_iterator()
  batch_labels, batch_ids, batch_values = train_dataset_iterator.get_next()

  validation_filename_list = [
      filename for filename in FLAGS.validation_files.split(",")
  ]
  validation_filename_placeholder = tf.placeholder(tf.string, shape=[None])
  validation_dataset = tf.data.TFRecordDataset(validation_filename_placeholder)
  validation_dataset = validation_dataset.map(parse_tfrecords_function).repeat(
  ).batch(FLAGS.validation_batch_size).shuffle(
      buffer_size=validation_buffer_size)
  validation_dataset_iterator = validation_dataset.make_initializable_iterator(
  )
  validation_labels, validation_ids, validation_values = validation_dataset_iterator.get_next(
  )

  # Define the model
  logits = inference(batch_ids, batch_values, True)
  batch_labels = tf.to_int64(batch_labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=batch_labels)
  loss = tf.reduce_mean(cross_entropy, name="loss")
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
  optimizer = util.get_optimizer_by_name(FLAGS.optimizer, learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)
  tf.get_variable_scope().reuse_variables()

  # Define accuracy op for train data
  train_accuracy_logits = inference(batch_ids, batch_values, False)
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
  outshape = tf.stack([derived_size, FLAGS.label_size])
  new_train_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
  _, train_auc = tf.contrib.metrics.streaming_auc(train_softmax,
                                                  new_train_batch_labels)

  # Define accuracy op for validate data
  validate_accuracy_logits = inference(validation_ids, validation_values,
                                       False)
  validate_softmax = tf.nn.softmax(validate_accuracy_logits)
  validate_batch_labels = tf.to_int64(validation_labels)
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
  outshape = tf.stack([derived_size, FLAGS.label_size])
  new_validate_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
  _, validate_auc = tf.contrib.metrics.streaming_auc(validate_softmax,
                                                     new_validate_batch_labels)

  # Define inference op
  sparse_index = tf.placeholder(tf.int64, [None, 2])
  sparse_ids = tf.placeholder(tf.int64, [None])
  sparse_values = tf.placeholder(tf.float32, [None])
  sparse_shape = tf.placeholder(tf.int64, [2])
  inference_ids = tf.SparseTensor(sparse_index, sparse_ids, sparse_shape)
  inference_values = tf.SparseTensor(sparse_index, sparse_values, sparse_shape)
  inference_logits = inference(inference_ids, inference_values, False)
  inference_softmax = tf.nn.softmax(inference_logits)
  inference_op = tf.argmax(inference_softmax, 1)
  keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
  keys = tf.identity(keys_placeholder)

  signature_def_map = {
      signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
      signature_def_utils.build_signature_def(
          inputs={
              "keys": utils.build_tensor_info(keys_placeholder),
              "indexs": utils.build_tensor_info(sparse_index),
              "ids": utils.build_tensor_info(sparse_ids),
              "values": utils.build_tensor_info(sparse_values),
              "shape": utils.build_tensor_info(sparse_shape)
          },
          outputs={
              "keys": utils.build_tensor_info(keys),
              "softmax": utils.build_tensor_info(inference_softmax),
              "prediction": utils.build_tensor_info(inference_op)
          },
          method_name=signature_constants.PREDICT_METHOD_NAME)
  }

  # Initialize saver and summary
  saver = tf.train.Saver()
  tf.summary.scalar("loss", loss)
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
    writer = tf.summary.FileWriter(FLAGS.output_path, sess.graph)
    sess.run(init_op)
    sess.run(
        train_dataset_iterator.initializer,
        feed_dict={train_filename_placeholder: train_filename_list})
    sess.run(
        validation_dataset_iterator.initializer,
        feed_dict={validation_filename_placeholder: validation_filename_list})

    if FLAGS.mode == "train":
      # Restore session and start queue runner
      util.restore_from_checkpoint(sess, saver, latest_checkpoint_file_path)
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
              loss_value, train_accuracy_value, train_auc_value, validate_accuracy_value, auc_value, summary_value = sess.run(
                  [
                      loss, train_accuracy, train_auc, validate_accuracy,
                      validate_auc, summary_op
                  ])
              end_time = datetime.datetime.now()

              logging.info(
                  "[{}] Step: {}, loss: {}, train_acc: {}, train_auc: {}, valid_acc: {}, valid_auc: {}".
                  format(end_time - start_time, step, loss_value,
                         train_accuracy_value, train_auc_value,
                         validate_accuracy_value, auc_value))
              writer.add_summary(summary_value, step)
              saver.save(sess, checkpoint_file_path, global_step=step)
              start_time = end_time
      except tf.errors.OutOfRangeError:
        if FLAGS.benchmark_mode:
          print("Finish training for benchmark")
          exit(0)
        else:
          # Export the model after training
          util.save_model(
              FLAGS.model_path,
              FLAGS.model_version,
              sess,
              signature_def_map,
              is_save_graph=False)
      finally:
        coord.request_stop()
      coord.join(threads)

    elif FLAGS.mode == "save_model":
      if not util.restore_from_checkpoint(sess, saver,
                                          latest_checkpoint_file_path):
        logging.error("No checkpoint found, exit now")
        exit(1)

      util.save_model(
          FLAGS.model_path,
          FLAGS.model_version,
          sess,
          signature_def_map,
          is_save_graph=False)

    elif FLAGS.mode == "inference":
      if not util.restore_from_checkpoint(sess, saver,
                                          latest_checkpoint_file_path):
        logging.error("No checkpoint found, exit now")
        exit(1)

      # Load inference test data
      inference_result_file_name = "./inference_result.txt"
      inference_test_file_name = "./data/a8a_test.libsvm"
      labels = []
      feature_ids = []
      feature_values = []
      feature_index = []
      ins_num = 0
      for line in open(inference_test_file_name, "r"):
        tokens = line.split(" ")
        labels.append(int(tokens[0]))
        feature_num = 0
        for feature in tokens[1:]:
          feature_id, feature_value = feature.split(":")
          feature_ids.append(int(feature_id))
          feature_values.append(float(feature_value))
          feature_index.append([ins_num, feature_num])
          feature_num += 1
        ins_num += 1

      # Run inference
      start_time = datetime.datetime.now()
      prediction, prediction_softmax = sess.run(
          [inference_op, inference_softmax],
          feed_dict={
              sparse_index: feature_index,
              sparse_ids: feature_ids,
              sparse_values: feature_values,
              sparse_shape: [ins_num, FLAGS.feature_size]
          })

      end_time = datetime.datetime.now()

      # Compute accuracy
      label_number = len(labels)
      correct_label_number = 0
      for i in range(label_number):
        if labels[i] == prediction[i]:
          correct_label_number += 1
      accuracy = float(correct_label_number) / label_number

      # Compute auc
      expected_labels = np.array(labels)
      predict_labels = prediction_softmax[:, 0]
      fpr, tpr, thresholds = metrics.roc_curve(
          expected_labels, predict_labels, pos_label=0)
      auc = metrics.auc(fpr, tpr)
      logging.info("[{}] Inference accuracy: {}, auc: {}".format(
          end_time - start_time, accuracy, auc))

      # Save result into the file
      np.savetxt(inference_result_file_name, prediction_softmax, delimiter=",")
      logging.info(
          "Save result to file: {}".format(inference_result_file_name))

    elif FLAGS.mode == "inference_with_tfrecords":
      if not util.restore_from_checkpoint(sess, saver,
                                          latest_checkpoint_file_path):
        logging.error("No checkpoint found, exit now")
        exit(1)

      # Load inference test data
      inference_result_file_name = "./inference_result.txt"
      inference_test_file_name = "./data/a8a/a8a_test.libsvm.tfrecords"

      batch_feature_index = []
      batch_labels = []
      batch_ids = []
      batch_values = []
      ins_num = 0

      # Read from TFRecords files
      for serialized_example in tf.python_io.tf_record_iterator(
          inference_test_file_name):
        # Get serialized example from file
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        label = example.features.feature["label"].float_list.value
        ids = example.features.feature["ids"].int64_list.value
        values = example.features.feature["values"].float_list.value
        #print("label: {}, features: {}".format(label, " ".join([str(id) + ":" + str(value) for id, value in zip(ids, values)])))
        batch_labels.append(label)
        # Notice that using extend() instead of append() to flatten the values
        batch_ids.extend(ids)
        batch_values.extend(values)
        for i in xrange(len(ids)):
          batch_feature_index.append([ins_num, i])

        ins_num += 1

      # Run inference
      start_time = datetime.datetime.now()
      prediction, prediction_softmax = sess.run(
          [inference_op, inference_softmax],
          feed_dict={
              sparse_index: batch_feature_index,
              sparse_ids: batch_ids,
              sparse_values: batch_values,
              sparse_shape: [ins_num, FLAGS.feature_size]
          })

      end_time = datetime.datetime.now()

      # Compute accuracy
      label_number = len(batch_labels)
      correct_label_number = 0
      for i in range(label_number):
        if batch_labels[i] == prediction[i]:
          correct_label_number += 1
      accuracy = float(correct_label_number) / label_number

      # Compute auc
      expected_labels = np.array(batch_labels)
      predict_labels = prediction_softmax[:, 0]
      fpr, tpr, thresholds = metrics.roc_curve(
          expected_labels, predict_labels, pos_label=0)
      auc = metrics.auc(fpr, tpr)
      logging.info("[{}] Inference accuracy: {}, auc: {}".format(
          end_time - start_time, accuracy, auc))

      # Save result into the file
      np.savetxt(inference_result_file_name, prediction_softmax, delimiter=",")
      logging.info(
          "Save result to file: {}".format(inference_result_file_name))


if __name__ == "__main__":
  main()
