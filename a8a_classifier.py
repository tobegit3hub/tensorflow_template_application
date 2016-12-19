#!/usr/bin/env python

import datetime
import json
import math
import numpy as np
import os

import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', None, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 1024,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_integer("validate_batch_size", 1024,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_integer("thread_number", 1, "Number of thread to read data")
flags.DEFINE_integer("min_after_dequeue", 100,
                     "indicates min_after_dequeue of shuffle queue")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                    "indicates the checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("model", "dnn",
                    "Model to train, option model: dnn, lr, wide_and_deep")
flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization or not")
flags.DEFINE_float('bn_epsilon', 0.001, 'The epsilon of batch normalization.')
flags.DEFINE_boolean("enable_dropout", False, "Enable dropout or not")
flags.DEFINE_float("dropout_keep_prob", 0.5, "The dropout keep prob")
flags.DEFINE_string("optimizer", "adagrad", "optimizer to train")
flags.DEFINE_integer('steps_to_validate', 10,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train", "Option mode: train, inference")
flags.DEFINE_string("model_path", "./model/", "indicates training output")
flags.DEFINE_integer("export_version", 1, "Version number of the model")

FEATURE_SIZE = 124
LABEL_SIZE = 2
TRAIN_TFRECORDS_FILE = "data/a8a_train.libsvm.tfrecords"
VALIDATE_TFRECORDS_FILE = "data/a8a_test.libsvm.tfrecords"

learning_rate = FLAGS.learning_rate
epoch_number = FLAGS.epoch_number
thread_number = FLAGS.thread_number
batch_size = FLAGS.batch_size
validate_batch_size = FLAGS.validate_batch_size
min_after_dequeue = FLAGS.min_after_dequeue
capacity = thread_number * batch_size + min_after_dequeue
mode = FLAGS.mode
checkpoint_dir = FLAGS.checkpoint_dir
if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)
tensorboard_dir = FLAGS.tensorboard_dir
if not os.path.exists(tensorboard_dir):
  os.makedirs(tensorboard_dir)


# Read TFRecords examples from filename queue
def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  return serialized_example

# Read TFRecords files for training
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(TRAIN_TFRECORDS_FILE),
    num_epochs=epoch_number)
serialized_example = read_and_decode(filename_queue)
batch_serialized_example = tf.train.shuffle_batch(
    [serialized_example],
    batch_size=batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)
features = tf.parse_example(batch_serialized_example,
                            features={
                                "label": tf.FixedLenFeature([], tf.float32),
                                "ids": tf.VarLenFeature(tf.int64),
                                "values": tf.VarLenFeature(tf.float32),
                            })
batch_labels = features["label"]
batch_ids = features["ids"]
batch_values = features["values"]

# Read TFRecords file for validatioin
validate_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(VALIDATE_TFRECORDS_FILE),
    num_epochs=epoch_number)
validate_serialized_example = read_and_decode(validate_filename_queue)
validate_batch_serialized_example = tf.train.shuffle_batch(
    [validate_serialized_example],
    batch_size=validate_batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)
validate_features = tf.parse_example(
    validate_batch_serialized_example,
    features={
        "label": tf.FixedLenFeature([], tf.float32),
        "ids": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32),
    })
validate_batch_labels = validate_features["label"]
validate_batch_ids = validate_features["ids"]
validate_batch_values = validate_features["values"]

# Define the model
input_units = FEATURE_SIZE
hidden1_units = 10
hidden2_units = 10
hidden3_units = 10
hidden4_units = 10
output_units = LABEL_SIZE


def full_connect(inputs, weights_shape, biases_shape):
  with tf.device('/cpu:0'):
    weights = tf.get_variable("weights",
                              weights_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases",
                             biases_shape,
                             initializer=tf.random_normal_initializer())
    layer = tf.matmul(inputs, weights) + biases

    if FLAGS.enable_bn:
      mean, var = tf.nn.moments(layer, axes=[0])
      scale = tf.get_variable("scale",
                              biases_shape,
                              initializer=tf.random_normal_initializer())
      shift = tf.get_variable("shift",
                              biases_shape,
                              initializer=tf.random_normal_initializer())
      layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                        FLAGS.bn_epsilon)
  return layer


def sparse_full_connect(sparse_ids, sparse_values, weights_shape,
                        biases_shape):
  with tf.device('/cpu:0'):
    weights = tf.get_variable("weights",
                              weights_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases",
                             biases_shape,
                             initializer=tf.random_normal_initializer())
  return tf.nn.embedding_lookup_sparse(
      weights, sparse_ids, sparse_values,
      combiner="sum") + biases


def full_connect_relu(inputs, weights_shape, biases_shape):
  return tf.nn.relu(full_connect(inputs, weights_shape, biases_shape))


def deep_inference(sparse_ids, sparse_values):
  with tf.variable_scope("layer1"):
    sparse_layer = sparse_full_connect(sparse_ids, sparse_values,
                                       [input_units, hidden1_units],
                                       [hidden1_units])
    layer = tf.nn.relu(sparse_layer)
  with tf.variable_scope("layer2"):
    layer = full_connect_relu(layer, [hidden1_units, hidden2_units],
                              [hidden2_units])
  with tf.variable_scope("layer3"):
    layer = full_connect_relu(layer, [hidden2_units, hidden3_units],
                              [hidden3_units])
  with tf.variable_scope("layer4"):
    layer = full_connect_relu(layer, [hidden3_units, hidden4_units],
                              [hidden4_units])

  if FLAGS.enable_dropout and FLAGS.mode == "train":
    layer = tf.nn.dropout(layer, FLAGS.dropout_keep_prob)

  with tf.variable_scope("output"):
    layer = full_connect(layer, [hidden4_units, output_units], [output_units])
  return layer


def wide_inference(sparse_ids, sparse_values):
  """
    Logistic regression model.
    """
  with tf.variable_scope("logistic_regression"):
    layer = sparse_full_connect(sparse_ids, sparse_values,
                                [input_units, output_units], [output_units])
  return layer


def wide_and_deep_inference(sparse_ids, sparse_values):
  return wide_inference(sparse_ids, sparse_values) + deep_inference(
      sparse_ids, sparse_values)


def inference(sparse_ids, sparse_values):
  print("Use the model: {}".format(FLAGS.model))
  if FLAGS.model == "lr":
    return wide_inference(sparse_ids, sparse_values)
  elif FLAGS.model == "dnn":
    return deep_inference(sparse_ids, sparse_values)
  elif FLAGS.model == "wide_and_deep":
    return wide_and_deep_inference(sparse_ids, sparse_values)
  else:
    print("Unknown model, exit now")
    exit(1)


logits = inference(batch_ids, batch_values)
batch_labels = tf.to_int64(batch_labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                               batch_labels)
loss = tf.reduce_mean(cross_entropy, name='loss')

print("Use the optimizer: {}".format(FLAGS.optimizer))
if FLAGS.optimizer == "sgd":
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
elif FLAGS.optimizer == "momentum":
  # optimizer = tf.train.MomentumOptimizer(learning_rate)
  print("Not support optimizer: {} yet, exit now".format(FLAGS.optimizer))
  exit(1)
elif FLAGS.optimizer == "adadelta":
  optimizer = tf.train.AdadeltaOptimizer(learning_rate)
elif FLAGS.optimizer == "adagrad":
  optimizer = tf.train.AdagradOptimizer(learning_rate)
elif FLAGS.optimizer == "adam":
  optimizer = tf.train.AdamOptimizer(learning_rate)
elif FLAGS.optimizer == "ftrl":
  optimizer = tf.train.FtrlOptimizer(learning_rate)
elif FLAGS.optimizer == "rmsprop":
  optimizer = tf.train.RMSPropOptimizer(learning_rate)
else:
  print("Unknow optimizer: {}, exit now".format(FLAGS.optimizer))
  exit(1)

with tf.device("/cpu:0"):
  global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

# Compute accuracy
tf.get_variable_scope().reuse_variables()
validate_accuracy_logits = inference(validate_batch_ids, validate_batch_values)
validate_softmax = tf.nn.softmax(validate_accuracy_logits)
validate_batch_labels = tf.to_int64(validate_batch_labels)
validate_correct_prediction = tf.equal(
    tf.argmax(validate_softmax, 1), validate_batch_labels)
validate_accuracy = tf.reduce_mean(tf.cast(validate_correct_prediction,
                                           tf.float32))

train_accuracy_logits = inference(batch_ids, batch_values)
train_softmax = tf.nn.softmax(train_accuracy_logits)
train_correct_prediction = tf.equal(tf.argmax(train_softmax, 1), batch_labels)
train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

# Compute auc
validate_batch_labels = tf.cast(validate_batch_labels, tf.int32)
sparse_labels = tf.reshape(validate_batch_labels, [-1, 1])
derived_size = tf.shape(validate_batch_labels)[0]
indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
concated = tf.concat(1, [indices, sparse_labels])
outshape = tf.pack([derived_size, LABEL_SIZE])
new_validate_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
_, auc_op = tf.contrib.metrics.streaming_auc(validate_softmax,
                                             new_validate_batch_labels)

# Define inference op
sparse_index = tf.placeholder(tf.int64)
sparse_ids = tf.placeholder(tf.int64)
sparse_values = tf.placeholder(tf.float32)
sparse_shape = tf.placeholder(tf.int64)
inference_ids = tf.SparseTensor(sparse_index, sparse_ids, sparse_shape)
inference_values = tf.SparseTensor(sparse_index, sparse_values, sparse_shape)
inference_logits = inference(inference_ids, inference_values)
inference_softmax = tf.nn.softmax(inference_logits)
inference_op = tf.argmax(inference_softmax, 1)

# Initialize saver and summary
checkpoint_file = checkpoint_dir + "/checkpoint.ckpt"
steps_to_validate = FLAGS.steps_to_validate
init_op = tf.initialize_all_variables()
tf.scalar_summary('loss', loss)
tf.scalar_summary('accuracy', validate_accuracy)
tf.scalar_summary('auc', auc_op)
saver = tf.train.Saver()
keys_placeholder = tf.placeholder("float")
keys = tf.identity(keys_placeholder)

# Create session to run graph
with tf.Session() as sess:
  summary_op = tf.merge_all_summaries()
  writer = tf.train.SummaryWriter(tensorboard_dir, sess.graph)
  sess.run(init_op)
  sess.run(tf.initialize_local_variables())

  if mode == "train":
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Continue training from the model {}".format(
          ckpt.model_checkpoint_path))
      saver.restore(sess, ckpt.model_checkpoint_path)

    # Get coordinator and run queues to read data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    start_time = datetime.datetime.now()
    try:
      while not coord.should_stop():
        _, loss_value, step = sess.run([train_op, loss, global_step])

        if step % steps_to_validate == 0:
          train_accuracy_value, validate_accuracy_value, auc_value, summary_value = sess.run(
              [train_accuracy, validate_accuracy, auc_op, summary_op])
          end_time = datetime.datetime.now()
          print(
              "[{}] Step: {}, loss: {}, train accuracy: {}, validate accuracy: {}, auc: {}".format(
                  end_time - start_time, step, loss_value,
                  train_accuracy_value, validate_accuracy_value, auc_value))

          writer.add_summary(summary_value, step)
          saver.save(sess, checkpoint_file, global_step=step)
          start_time = end_time
    except tf.errors.OutOfRangeError:
      print("Done training after reading all data")

      print("Exporting trained model to {}".format(FLAGS.model_path))
      model_exporter = exporter.Exporter(saver)
      model_exporter.init(
          sess.graph.as_graph_def(),
          named_graph_signatures={
              'inputs': exporter.generic_signature({"keys": keys_placeholder,
                                                    "index": sparse_index,
                                                    "ids": sparse_ids,
                                                    "values": sparse_values,
                                                    "shape": sparse_shape}),
              'outputs':
              exporter.generic_signature({"keys": keys,
                                          "softmax": inference_softmax,
                                          "prediction": inference_op})
          })
      model_exporter.export(FLAGS.model_path,
                            tf.constant(FLAGS.export_version), sess)
      print 'Done exporting!'

    finally:
      coord.request_stop()

    # Wait for threads to exit
    coord.join(threads)

  elif mode == "inference":
    print("Start to run inference")
    start_time = datetime.datetime.now()

    inference_result_file_name = "./a8a_test_result.libsvm"
    inference_test_file_name = "./data/a8a_test.libsvm"
    feature_ids = []
    feature_values = []
    feature_index = []
    ins_num = 0
    for line in open(inference_test_file_name, "r"):
      tokens = line.split(" ")
      feature_num = 0
      for feature in tokens[1:]:
        feature_id, feature_value = feature.split(":")
        feature_ids.append(int(feature_id))
        feature_values.append(float(feature_value))
        feature_index.append([ins_num, feature_num])
        feature_num += 1
      ins_num += 1

    # Restore wights from model file
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Use the model {}".format(ckpt.model_checkpoint_path))
      saver.restore(sess, ckpt.model_checkpoint_path)
      inference_result = sess.run(inference_softmax,
                                  feed_dict={sparse_index: feature_index,
                                             sparse_ids: feature_ids,
                                             sparse_values: feature_value,
                                             sparse_shape:
                                             [ins_num, FEATURE_SIZE]})

      end_time = datetime.datetime.now()
      print("[{}] Inference result: {}".format(end_time - start_time,
                                               inference_result))
      np.savetxt(inference_result_file_name, inference_result, delimiter=",")
      print("Save result to file: {}".format(inference_result_file_name))

    else:
      print("No model found, exit now")
      exit(1)
