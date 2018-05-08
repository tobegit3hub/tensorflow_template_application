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
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)

import model


def define_flags():
  """
  Define all the command-line parameters.
  
  Return:
    The FLAGS object.
  """

  flags = tf.app.flags
  flags.DEFINE_string("mode", "train", "Support train, inference, savedmodel")
  flags.DEFINE_boolean("enable_benchmark", False, "Enable benchmark")
  flags.DEFINE_boolean("enable_colored_log", False, "Enable colored log")
  flags.DEFINE_boolean("resume_from_checkpoint", True, "Resume or not")
  flags.DEFINE_string("scenario", "classification",
                      "Support classification, regression")
  flags.DEFINE_integer("feature_size", 9, "Number of feature size")
  flags.DEFINE_integer("label_size", 2, "Number of label size")
  flags.DEFINE_string("train_file_format", "tfrecords",
                      "Support tfrecords, csv")
  flags.DEFINE_string("train_file", "./data/cancer/cancer_train.csv.tfrecords",
                      "Train files which supports glob pattern")
  flags.DEFINE_string("validation_file",
                      "./data/cancer/cancer_test.csv.tfrecords",
                      "Validate files which supports glob pattern")
  flags.DEFINE_string("inference_data_file", "./data/cancer/cancer_test.csv",
                      "Data file for inference")
  flags.DEFINE_string("inference_result_file", "./inference_result.txt",
                      "Result file from inference")
  flags.DEFINE_string("optimizer", "adagrad",
                      "Support sgd, adadelta, adagrad, adam, ftrl, rmsprop")
  flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
  flags.DEFINE_string("model", "dnn",
                      "Support dnn, lr, wide_and_deep, customized, cnn")
  flags.DEFINE_string("dnn_struct", "128 32 8", "DNN struct")
  flags.DEFINE_integer("epoch_number", 100, "Number of epoches")
  flags.DEFINE_integer("batch_size", 1024, "Batch size")
  flags.DEFINE_integer("validate_batch_size", 1024,
                       "Batch size for validation")
  flags.DEFINE_integer("batch_thread_number", 1, "Batch thread number")
  flags.DEFINE_integer("min_after_dequeue", 100, "Min after dequeue")
  flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization")
  flags.DEFINE_float("bn_epsilon", 0.001, "Epsilon of batch normalization")
  flags.DEFINE_boolean("enable_dropout", False, "Enable dropout")
  flags.DEFINE_float("dropout_keep_prob", 0.5, "Keep prob of dropout")
  flags.DEFINE_boolean("enable_lr_decay", False, "Enable learning rate decay")
  flags.DEFINE_float("lr_decay_rate", 0.96, "Learning rate decay rate")
  flags.DEFINE_integer("steps_to_validate", 10, "Steps to validate")
  flags.DEFINE_string("checkpoint_path", "./checkpoint/",
                      "Path for checkpoint")
  flags.DEFINE_string("output_path", "./tensorboard/", "Path for tensorboard")
  flags.DEFINE_string("model_path", "./model/", "Path of the model")
  flags.DEFINE_integer("model_version", 1, "Version of the model")
  FLAGS = flags.FLAGS

  # Check parameters
  assert (FLAGS.mode in ["train", "inference", "savedmodel"])
  assert (FLAGS.scenario in ["classification", "regression"])
  assert (FLAGS.train_file_format in ["tfrecords", "csv"])
  assert (FLAGS.optimizer in [
      "sgd", "adadelta", "adagrad", "adam", "ftrl", "rmsprop"
  ])
  assert (FLAGS.model in ["dnn", "lr", "wide_and_deep", "customized", "cnn"])

  # Print flags
  parameter_value_map = {}
  for key in FLAGS.__flags.keys():
    parameter_value_map[key] = FLAGS.__flags[key].value
  pprint.PrettyPrinter().pprint(parameter_value_map)

  return FLAGS


def get_optimizer_by_name(optimizer_name, learning_rate):
  """
  Get optimizer object by the string.
  
  Args:
    optimizer_name: Name of the optimizer. 
    learning_rate: The learning rate.
  
  Return:
    The optimizer object.
  """

  logging.info("Use the optimizer: {}".format(optimizer_name))
  if optimizer_name == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer_name == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif optimizer_name == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif optimizer_name == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif optimizer_name == "ftrl":
    optimizer = tf.train.FtrlOptimizer(learning_rate)
  elif optimizer_name == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
  else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  return optimizer


def save_model(model_path,
               model_version,
               sess,
               model_signature,
               is_save_graph=False):
  """
  Save the model in standard SavedModel format.
  
  Args:
    model_path: The path to model.
    model_version: The version of model.
    sess: The TensorFlow Session object.
    model_signature: The TensorFlow SignatureDef object.
    is_save_graph: Should save graph file of not.
  
  Return:
    None
  """

  export_path = os.path.join(model_path, str(model_version))

  if os.path.isdir(export_path) == True:
    logging.error("The model exists in path: {}".format(export_path))
    return

  try:
    # Save the SavedModel
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        clear_devices=True,
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            model_signature,
        },
        legacy_init_op=legacy_init_op)
    logging.info("Save the model in: {}".format(export_path))
    builder.save()

    # Save the GraphDef
    if is_save_graph == True:
      graph_file_name = "graph.pb"
      logging.info("Save the graph file in: {}".format(model_path))
      tf.train.write_graph(
          sess.graph_def, model_path, graph_file_name, as_text=False)

  except Exception as e:
    logging.error("Fail to export saved model, exception: {}".format(e))


def restore_from_checkpoint(sess, saver, checkpoint_file_path):
  """
  Restore session from checkpoint files.
  
  Args:
    sess: TensorFlow Session object.
    saver: TensorFlow Saver object.
    checkpoint_file_path: The checkpoint file path.
  
  Return:
    True if restore successfully and False if fail
  """
  if checkpoint_file_path:
    logging.info(
        "Restore session from checkpoint: {}".format(checkpoint_file_path))
    saver.restore(sess, checkpoint_file_path)
    return True
  else:
    logging.error("Checkpoint not found: {}".format(checkpoint_file_path))
    return False


def parse_tfrecords_function(example_proto):
  """
  Decode TFRecords for Dataset.
  
  Args:
    example_proto: TensorFlow ExampleProto object. 
  
  Return:
    The op of features and labels
  """
  features = {
      "features": tf.FixedLenFeature([FLAGS.feature_size], tf.float32),
      "label": tf.FixedLenFeature([], tf.int64, default_value=0)
  }
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["features"], parsed_features["label"]


# TODO: Change for dataset api
def read_and_decode_csv(filename_queue):
  # Notice that it supports label in the last column only
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)
  record_defaults = [[1.0] for i in range(FLAGS.feature_size)] + [[0]]
  columns = tf.decode_csv(value, record_defaults=record_defaults)
  label = columns[-1]
  features = tf.stack(columns[0:-1])
  return label, features


def inference(inputs, input_units, output_units, is_train=True):
  """
  Define the model by model name.
  
  Return:
    The logit of the model output.
  """

  if FLAGS.model == "dnn":
    return model.dnn_inference(inputs, input_units, output_units, is_train,
                               FLAGS)
  elif FLAGS.model == "lr":
    return model.lr_inference(inputs, input_units, output_units, is_train,
                              FLAGS)
  elif FLAGS.model == "wide_and_deep":
    return model.wide_and_deep_inference(inputs, input_units, output_units,
                                         is_train, FLAGS)
  elif FLAGS.model == "customized":
    return model.customized_inference(inputs, input_units, output_units,
                                      is_train, FLAGS)
  elif FLAGS.model == "cnn":
    return model.cnn_inference(inputs, input_units, output_units, is_train,
                               FLAGS)


logging.basicConfig(level=logging.INFO)
FLAGS = define_flags()
if FLAGS.enable_colored_log:
  import coloredlogs
  coloredlogs.install()


def main():
  # Get hyper-parameters
  if os.path.exists(FLAGS.checkpoint_path) == False:
    os.makedirs(FLAGS.checkpoint_path)
  CHECKPOINT_FILE = FLAGS.checkpoint_path + "/checkpoint.ckpt"
  LATEST_CHECKPOINT = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

  if os.path.exists(FLAGS.output_path) == False:
    os.makedirs(FLAGS.output_path)

  batch_size = 10
  buffer_size = 100

  # Step 1: Construct the dataset op
  epoch_number = FLAGS.epoch_number
  if epoch_number <= 0:
    epoch_number = -1

  train_filename_list = [FLAGS.train_file]
  train_filename_placeholder = tf.placeholder(tf.string, shape=[None])
  train_dataset = tf.data.TFRecordDataset(train_filename_placeholder)
  train_dataset = train_dataset.map(parse_tfrecords_function).repeat(
      epoch_number).batch(batch_size).shuffle(buffer_size=buffer_size)
  train_dataset_iterator = train_dataset.make_initializable_iterator()
  batch_features_op, batch_label_op = train_dataset_iterator.get_next()
  batch_label_op = tf.cast(batch_label_op, tf.int32)

  validation_filename_list = [FLAGS.validation_file]
  validation_filename_placeholder = tf.placeholder(tf.string, shape=[None])
  validation_dataset = tf.data.TFRecordDataset(validation_filename_placeholder)
  validation_dataset = validation_dataset.map(
      parse_tfrecords_function).repeat(epoch_number).batch(
          batch_size).shuffle(buffer_size=buffer_size)
  validation_dataset_iterator = validation_dataset.make_initializable_iterator(
  )
  validate_batch_features, validate_batch_labels = validation_dataset_iterator.get_next(
  )
  validate_batch_labels = tf.cast(validate_batch_labels, tf.int32)

  """
  if FLAGS.train_file_format == "tfrecords":
    pass
    #read_and_decode_function = read_and_decode_tfrecords
  elif FLAGS.train_file_format == "csv":
    pass
    #read_and_decode_function = read_and_decode_csv
  """

  # Step 2: Define the model
  input_units = FLAGS.feature_size
  output_units = FLAGS.label_size
  logits = inference(batch_features_op, input_units, output_units, True)

  if FLAGS.scenario == "classification":
    batch_label_op = tf.to_int64(batch_label_op)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #logits=logits, labels=batch_labels)
        logits=logits,
        labels=batch_label_op)
    loss = tf.reduce_mean(cross_entropy, name="loss")
  elif FLAGS.scenario == "regression":
    msl = tf.square(logits - batch_label_op, name="msl")
    loss = tf.reduce_mean(msl, name="loss")

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
  optimizer = get_optimizer_by_name(FLAGS.optimizer, learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)
  tf.get_variable_scope().reuse_variables()

  # Avoid error when not using acc and auc op
  if FLAGS.scenario == "regression":
    batch_labels = tf.to_int64(batch_label_op)

  # Define accuracy op for train data
  train_accuracy_logits = inference(batch_features_op, input_units,
                                    output_units, False)
  train_softmax = tf.nn.softmax(train_accuracy_logits)
  train_correct_prediction = tf.equal(
      tf.argmax(train_softmax, 1), batch_label_op)
  train_accuracy = tf.reduce_mean(
      tf.cast(train_correct_prediction, tf.float32))

  # Define auc op for train data
  batch_labels = tf.cast(batch_label_op, tf.int32)
  sparse_labels = tf.reshape(batch_labels, [-1, 1])
  derived_size = tf.shape(batch_labels)[0]
  indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
  concated = tf.concat(axis=1, values=[indices, sparse_labels])
  outshape = tf.stack([derived_size, FLAGS.label_size])
  new_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
  _, train_auc = tf.contrib.metrics.streaming_auc(train_softmax,
                                                  new_batch_labels)

  # Define accuracy op for validate data
  validate_accuracy_logits = inference(validate_batch_features, input_units,
                                       output_units, False)
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
  outshape = tf.stack([derived_size, FLAGS.label_size])
  new_validate_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
  _, validate_auc = tf.contrib.metrics.streaming_auc(validate_softmax,
                                                     new_validate_batch_labels)

  # Define inference op
  inference_features = tf.placeholder(
      "float", [None, FLAGS.feature_size], name="features")
  inference_logits = inference(inference_features, input_units, output_units,
                               False)
  inference_softmax = tf.nn.softmax(inference_logits, name="output_softmax")
  inference_op = tf.argmax(inference_softmax, 1, name="output_prediction")
  keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1], name="keys")
  keys_identity = tf.identity(keys_placeholder, name="output_keys")
  model_signature = signature_def_utils.build_signature_def(
      inputs={
          "keys": utils.build_tensor_info(keys_placeholder),
          "features": utils.build_tensor_info(inference_features)
      },
      outputs={
          "keys": utils.build_tensor_info(keys_identity),
          "prediction": utils.build_tensor_info(inference_op),
          "softmax": utils.build_tensor_info(inference_softmax),
      },
      method_name=signature_constants.PREDICT_METHOD_NAME)

  # Initialize saver and summary
  saver = tf.train.Saver()
  tf.summary.scalar("loss", loss)
  if FLAGS.scenario == "classification":
    tf.summary.scalar("train_accuracy", train_accuracy)
    tf.summary.scalar("train_auc", train_auc)
    tf.summary.scalar("validate_accuracy", validate_accuracy)
    tf.summary.scalar("validate_auc", validate_auc)
  summary_op = tf.summary.merge_all()
  init_op = [
      tf.global_variables_initializer(),
      tf.local_variables_initializer()
  ]

  # Step 3: Create session to run
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
      if FLAGS.resume_from_checkpoint:
        restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT)

      start_time = datetime.datetime.now()

      try:
        while True:
          if FLAGS.enable_benchmark:
            sess.run(train_op)
          else:
            _, global_step_value = sess.run([train_op, global_step])

            # Step 4: Print state while training
            if global_step_value % FLAGS.steps_to_validate == 0:
              if FLAGS.scenario == "classification":
                loss_value, train_accuracy_value, train_auc_value, validate_accuracy_value, validate_auc_value, summary_value = sess.run(
                    [
                        loss, train_accuracy, train_auc, validate_accuracy,
                        validate_auc, summary_op
                    ])
                end_time = datetime.datetime.now()

                logging.info(
                    "[{}] Step: {}, loss: {}, train_acc: {}, train_auc: {}, valid_acc: {}, valid_auc: {}".
                    format(end_time - start_time, global_step_value,
                           loss_value, train_accuracy_value, train_auc_value,
                           validate_accuracy_value, validate_auc_value))

              elif FLAGS.scenario == "regression":
                loss_value, summary_value = sess.run([loss, summary_op])
                end_time = datetime.datetime.now()
                logging.info("[{}] Step: {}, loss: {}".format(
                    end_time - start_time, global_step_value, loss_value))

              writer.add_summary(summary_value, global_step_value)
              saver.save(sess, CHECKPOINT_FILE, global_step=global_step_value)

              start_time = end_time
      except tf.errors.OutOfRangeError:
        if FLAGS.enable_benchmark:
          logging.info("Finish training for benchmark")
        else:
          # Step 5: Export the model after training
          save_model(
              FLAGS.model_path,
              FLAGS.model_version,
              sess,
              model_signature,
              is_save_graph=False)

    elif FLAGS.mode == "savedmodel":
      if restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT) == False:
        logging.error("No checkpoint for exporting model, exit now")
        return

      save_model(
          FLAGS.model_path,
          FLAGS.model_version,
          sess,
          model_signature,
          is_save_graph=False)

    elif FLAGS.mode == "inference":
      if restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT) == False:
        logging.error("No checkpoint for inference, exit now")
        return

      # Load inference test data
      inference_result_file_name = FLAGS.inference_result_file
      inference_test_file_name = FLAGS.inference_data_file
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


if __name__ == "__main__":
  main()
