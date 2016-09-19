#!/usr/bin/env python

import datetime
import json
import math
import numpy as np
import os
import tensorflow as tf

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
flags.DEFINE_string("model", "wide_and_deep",
                    "Model to train, option model: wide, deep, wide_and_deep")
flags.DEFINE_string("optimizer", "adagrad", "optimizer to train")
flags.DEFINE_integer('steps_to_validate', 100,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train",
                    "Option mode: train, train_from_scratch, inference")

FEATURE_SIZE = 9
LABEL_SIZE = 2
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
  features = tf.parse_single_example(
      serialized_example,
      features={
          "label": tf.FixedLenFeature([], tf.float32),
          "features": tf.FixedLenFeature([FEATURE_SIZE], tf.float32),
      })
  label = features["label"]
  features = features["features"]
  return label, features

# Read TFRecords files for training
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("data/cancer_train.csv.tfrecords"),
    num_epochs=epoch_number)
label, features = read_and_decode(filename_queue)
batch_labels, batch_features = tf.train.shuffle_batch(
    [label, features],
    batch_size=batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)

# Read TFRecords file for validatioin
validate_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("data/cancer_test.csv.tfrecords"),
    num_epochs=epoch_number)
validate_label, validate_features = read_and_decode(validate_filename_queue)
validate_batch_labels, validate_batch_features = tf.train.shuffle_batch(
    [validate_label, validate_features],
    batch_size=validate_batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)

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
  return tf.matmul(inputs, weights) + biases


def full_connect_relu(inputs, weights_shape, biases_shape):
  return tf.nn.relu(full_connect(inputs, weights_shape, biases_shape))


def deep_inference(inputs):
  '''
  Shape of inputs should be [batch_size, input_units], return shape [batch_size, output_units]
  '''
  with tf.variable_scope("layer1"):
    layer = full_connect_relu(inputs, [input_units, hidden1_units],
                              [hidden1_units])
  with tf.variable_scope("layer2"):
    layer = full_connect_relu(layer, [hidden1_units, hidden2_units],
                              [hidden2_units])
  with tf.variable_scope("layer3"):
    layer = full_connect_relu(layer, [hidden2_units, hidden3_units],
                              [hidden3_units])
  with tf.variable_scope("layer4"):
    layer = full_connect_relu(layer, [hidden3_units, hidden4_units],
                              [hidden4_units])
  with tf.variable_scope("outpu"):
    layer = full_connect(layer, [hidden4_units, output_units], [output_units])
  return layer


def wide_inference(inputs):
  """
  Logistic regression model.
  """
  with tf.variable_scope("logistic_regression"):
    layer = full_connect(inputs, [input_units, output_units], [output_units])
  return layer


def wide_and_deep_inference(inputs):
  return wide_inference(inputs) + deep_inference(inputs)


def inference(inputs):
  print("Use the model: {}".format(FLAGS.model))
  if FLAGS.model == "wide":
    return wide_inference(inputs)
  elif FLAGS.model == "deep":
    return deep_inference(inputs)
  elif FLAGS.model == "wide_and_deep":
    return wide_and_deep_inference(inputs)
  else:
    print("Unknown model, exit now")
    exit(1)


logits = inference(batch_features)
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
accuracy_logits = inference(validate_batch_features)
validate_softmax = tf.nn.softmax(accuracy_logits)
validate_batch_labels = tf.to_int64(validate_batch_labels)
correct_prediction = tf.equal(
    tf.argmax(validate_softmax, 1), validate_batch_labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
inference_features = tf.placeholder("float", [None, FEATURE_SIZE])
inference_logits = inference(inference_features)
inference_softmax = tf.nn.softmax(inference_logits)
inference_op = tf.argmax(inference_softmax, 1)

# Initialize saver and summary
checkpoint_file = checkpoint_dir + "/checkpoint.ckpt"
steps_to_validate = FLAGS.steps_to_validate
init_op = tf.initialize_all_variables()
tf.scalar_summary('loss', loss)
tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('auc', auc_op)
saver = tf.train.Saver()
keys_placeholder = tf.placeholder("float")
keys = tf.identity(keys_placeholder)
tf.add_to_collection("inputs", json.dumps({'key': keys_placeholder.name,
                                           'features':
                                           inference_features.name}))
tf.add_to_collection("outputs", json.dumps({'key': keys.name,
                                            'softmax': inference_softmax.name,
                                            'prediction': inference_op.name}))

# Create session to run graph
with tf.Session() as sess:
  summary_op = tf.merge_all_summaries()
  writer = tf.train.SummaryWriter(tensorboard_dir, sess.graph)
  sess.run(init_op)
  sess.run(tf.initialize_local_variables())

  if mode == "train" or mode == "train_from_scratch":
    if mode != "train_from_scratch":
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
          accuracy_value, auc_value, summary_value = sess.run(
              [accuracy, auc_op, summary_op])
          end_time = datetime.datetime.now()
          print("[{}] Step: {}, loss: {}, accuracy: {}, auc: {}".format(
              end_time - start_time, step, loss_value, accuracy_value,
              auc_value))

          writer.add_summary(summary_value, step)
          saver.save(sess, checkpoint_file, global_step=step)
          start_time = end_time
    except tf.errors.OutOfRangeError:
      print("Done training after reading all data")
    finally:
      coord.request_stop()

    # Wait for threads to exit
    coord.join(threads)

  elif mode == "inference":
    print("Start to run inference")

    '''
    inference_data = np.array(
        [(10, 10, 10, 8, 6, 1, 8, 9, 1), (6, 2, 1, 1, 1, 1, 7, 1, 1),
         (2, 5, 3, 3, 6, 7, 7, 5, 1), (10, 4, 3, 1, 3, 3, 6, 5, 2),
         (6, 10, 10, 2, 8, 10, 7, 3, 3), (5, 6, 5, 6, 10, 1, 3, 1, 1),
         (1, 1, 1, 1, 2, 1, 2, 1, 2), (3, 7, 7, 4, 4, 9, 4, 8, 1),
         (1, 1, 1, 1, 2, 1, 2, 1, 1), (4, 1, 1, 3, 2, 1, 3, 1, 1)])
    correct_labels = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0]
    '''

    inference_result_file_name = "./cancer_inference_result.csv"
    inference_test_file_name = "./data/cancer_inference.csv"
    inference_data = np.genfromtxt(inference_test_file_name, delimiter=',')

    # Restore wights from model file
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Use the model {}".format(ckpt.model_checkpoint_path))
      saver.restore(sess, ckpt.model_checkpoint_path)
      inference_result = sess.run(
          inference_op,
          # inference_softmax,
          feed_dict={inference_features: inference_data})
      print("Inference result: {}".format(inference_result))

    else:
      print("No model found, exit now")
      exit(1)
