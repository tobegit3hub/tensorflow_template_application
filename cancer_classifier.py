#!/usr/bin/env python

import tensorflow as tf
import math
import os
import numpy as np
import json

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', None, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 1024,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_integer("thread_number", 1, "Number of thread to read data")
flags.DEFINE_integer("min_after_dequeue", 100,
                     "indicates min_after_dequeue of shuffle queue")
flags.DEFINE_string("output_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("model", "deep",
                    "Model to train, option model: deep, linear")
flags.DEFINE_string("optimizer", "sgd", "optimizer to import")
flags.DEFINE_integer('hidden1', 10, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 20, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('steps_to_validate', 10,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train",
                    "Option mode: train, train_from_scratch, inference")

# Hyperparameter
learning_rate = FLAGS.learning_rate
epoch_number = FLAGS.epoch_number
thread_number = FLAGS.thread_number
batch_size = FLAGS.batch_size
min_after_dequeue = FLAGS.min_after_dequeue
capacity = thread_number * batch_size + min_after_dequeue
FEATURE_SIZE = 9


# Read serialized examples from filename queue
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
    tf.train.match_filenames_once("data/cancer.csv.tfrecords"),
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
    batch_size=batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)

# Define the model
input_units = FEATURE_SIZE
hidden1_units = 10
hidden2_units = 20
hidden3_units = 10
hidden4_units = 30
output_units = 2

def full_connect_relu(inputs, weights_shape, biases_shape):
    weights = tf.get_variable("weights", weights_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", biases_shape, initializer=tf.random_normal_initializer())
    return tf.nn.relu(tf.matmul(inputs, weights) + biases)

def full_connect(inputs, weights_shape, biases_shape):
    weights = tf.get_variable("weights", weights_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", biases_shape, initializer=tf.random_normal_initializer())
    return tf.matmul(inputs, weights) + biases

def inference():
    with tf.variable_scope("layer1"):
        layer = full_connect_relu(batch_features, [input_units, hidden1_units], [hidden1_units])
    with tf.variable_scope("layer2"):
        layer = full_connect_relu(layer, [hidden1_units, hidden2_units], [hidden2_units])
    with tf.variable_scope("layer3"):
        layer = full_connect_relu(layer, [hidden2_units, hidden3_units], [hidden3_units])
    with tf.variable_scope("layer4"):
        layer = full_connect_relu(layer, [hidden3_units, hidden4_units], [hidden4_units])
    with tf.variable_scope("outpu"):
        layer = full_connect(layer, [hidden4_units, output_units], [output_units])
    return layer

'''
# Hidden 1
weights1 = tf.Variable(tf.truncated_normal([input_units, hidden1_units]), dtype=tf.float32, name='weights')
biases1 = tf.Variable(tf.truncated_normal([hidden1_units]), name='biases', dtype=tf.float32)
hidden1 = tf.nn.relu(tf.matmul(batch_features, weights1) + biases1)

# Hidden 2
weights2 = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units]), dtype=tf.float32, name='weights')
biases2 = tf.Variable(tf.truncated_normal([hidden2_units]), name='biases', dtype=tf.float32)
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

# Linear
weights3 = tf.Variable(tf.truncated_normal([hidden2_units, output_units]), dtype=tf.float32, name='weights')
biases3 = tf.Variable(tf.truncated_normal([output_units]), name='biases', dtype=tf.float32)
logits = tf.matmul(hidden2, weights3) + biases3
'''


logits = inference()

batch_labels = tf.to_int64(batch_labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, batch_labels)
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
else:
    optimizer = tf.train.MomentumOptimizer(learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)


'''
# Compute accuracy
accuracy_hidden1 = tf.nn.relu(tf.matmul(validate_batch_features, weights1) +
                              biases1)
accuracy_hidden2 = tf.nn.relu(tf.matmul(accuracy_hidden1, weights2) + biases2)
accuracy_logits = tf.matmul(accuracy_hidden2, weights3) + biases3
validate_softmax = tf.nn.softmax(accuracy_logits)
validate_batch_labels = tf.to_int64(validate_batch_labels)
correct_prediction = tf.equal(
    tf.argmax(validate_softmax, 1), validate_batch_labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Compute auc
validate_batch_labels = tf.cast(validate_batch_labels, tf.int32)
num_labels = 2
sparse_labels = tf.reshape(validate_batch_labels, [-1, 1])
derived_size = tf.shape(validate_batch_labels)[0]
indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
concated = tf.concat(1, [indices, sparse_labels])
outshape = tf.pack([derived_size, num_labels])
new_validate_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
_, auc_op = tf.contrib.metrics.streaming_auc(validate_softmax,
                                             new_validate_batch_labels)

# Define inference op
inference_features = tf.placeholder("float", [None, FEATURE_SIZE])
inference_hidden1 = tf.nn.relu(tf.matmul(inference_features, weights1) +
                               biases1)
inference_hidden2 = tf.nn.relu(tf.matmul(inference_hidden1, weights2) +
                               biases2)
inference_logits = tf.matmul(inference_hidden2, weights3) + biases3
inference_softmax = tf.nn.softmax(inference_logits)
inference_op = tf.argmax(inference_softmax, 1)
'''

# Initialize saver and summary
mode = FLAGS.mode
steps_to_validate = FLAGS.steps_to_validate
init_op = tf.initialize_all_variables()
tf.scalar_summary('loss', loss)
'''
tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('auc', auc_op)
'''
saver = tf.train.Saver()
keys_placeholder = tf.placeholder("float")
keys = tf.identity(keys_placeholder)
'''
tf.add_to_collection("inputs", json.dumps({'key': keys_placeholder.name,
                                           'features':
                                           inference_features.name}))
tf.add_to_collection("outputs", json.dumps({'key': keys.name,
                                            'softmax': inference_softmax.name,
                                            'prediction': inference_op.name}))
'''

# Create session to run graph
with tf.Session() as sess:
    summary_op = tf.merge_all_summaries()
    output_dir = FLAGS.output_dir
    writer = tf.train.SummaryWriter(output_dir, sess.graph)
    sess.run(init_op)
    sess.run(tf.initialize_local_variables())

    if mode == "train" or mode == "train_from_scratch":
        if mode != "train_from_scratch":
            ckpt = tf.train.get_checkpoint_state("./checkpoint/")
            if ckpt and ckpt.model_checkpoint_path:
                print("Continue training from the model {}".format(
                    ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)

        # Get coordinator and run queues to read data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            while not coord.should_stop():
                _, loss_value, step = sess.run([train_op, loss, global_step])
                if step % steps_to_validate == 0:
                    print("Step: {}, loss: {}".format(step, loss_value))
                    '''
                    accuracy_value, auc_value, summary_value = sess.run(
                        [accuracy, auc_op, summary_op])
                    print("Step: {}, loss: {}, accuracy: {}, auc: {}".format(
                        step, loss_value, accuracy_value, auc_value))
                    writer.add_summary(summary_value, step)
                    saver.save(sess,
                               "./checkpoint/checkpoint.ckpt",
                               global_step=step)
                    '''
        except tf.errors.OutOfRangeError:
            print("Done training after reading all data")
        finally:
            coord.request_stop()

        # Wait for threads to exit
        coord.join(threads)

    elif mode == "inference":
        print("Start to run inference")

        inference_data = np.array(
            [(10, 10, 10, 8, 6, 1, 8, 9, 1), (6, 2, 1, 1, 1, 1, 7, 1, 1),
             (2, 5, 3, 3, 6, 7, 7, 5, 1), (10, 4, 3, 1, 3, 3, 6, 5, 2),
             (6, 10, 10, 2, 8, 10, 7, 3, 3), (5, 6, 5, 6, 10, 1, 3, 1, 1),
             (1, 1, 1, 1, 2, 1, 2, 1, 2), (3, 7, 7, 4, 4, 9, 4, 8, 1),
             (1, 1, 1, 1, 2, 1, 2, 1, 1), (4, 1, 1, 3, 2, 1, 3, 1, 1)])
        correct_labels = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0]

        # Restore wights from model file
        ckpt = tf.train.get_checkpoint_state("./checkpoint/")
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
