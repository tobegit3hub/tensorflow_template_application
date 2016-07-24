#!/usr/bin/env python

import tensorflow as tf
import math
import os

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', None, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 1024, "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_integer("thread_number", 1, "Number of thread to read data")
flags.DEFINE_integer("min_after_dequeue", 100, "indicates min_after_dequeue of shuffle queue")
flags.DEFINE_string("output_dir", "./tensorboard/", "indicates training output")
flags.DEFINE_string("optimizer", "sgd", "optimizer to import")
flags.DEFINE_integer('hidden1', 10, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 20, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('steps_to_validate', 10, 'Steps to validate and print loss')
flags.DEFINE_string("mode", "train", "Option mode: train, test, inference")


TRAIN_MODE = "train"
TEST_MODE = "test"
INFERENCE_MODE = "inference"

feature_size = 9
# Read serialized examples from filename queue
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.float32),
            "features": tf.FixedLenFeature([feature_size], tf.float32),
        })

    label = features["label"]
    features = features["features"]

    return label, features

# Read the TFRecords file
#current_path = os.getcwd()
#input_file = os.path.join(current_path, "data/part.tfrecords")

# Hyperparameter
learning_rate = FLAGS.learning_rate
epoch_number = FLAGS.epoch_number
thread_number = FLAGS.thread_number
batch_size = FLAGS.batch_size
min_after_dequeue = FLAGS.min_after_dequeue
capacity = thread_number * batch_size + min_after_dequeue

# Batch get label and features
#filename_queue = tf.train.string_input_producer([input_file],
#                                                num_epochs=epoch_number)
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("data/cancer.csv.tfrecords"), num_epochs=epoch_number)
label, features = read_and_decode(filename_queue)
batch_labels, batch_features = tf.train.shuffle_batch(
    [label, features],
    batch_size=batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)

validate_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("data/cancer_test.csv.tfrecords"), num_epochs=epoch_number)
validate_label, validate_features = read_and_decode(validate_filename_queue)
validate_batch_labels, validate_batch_features = tf.train.shuffle_batch(
    [validate_label, validate_features],
    batch_size=batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)
'''
# Define the model
input_units = feature_size
hidden1_units = FLAGS.hidden1
hidden2_units = FLAGS.hidden2
output_units = 2

# Hidden 1
with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([input_units, hidden1_units]), dtype=tf.float32, name='weights')
    biases = tf.Variable(tf.truncated_normal([hidden1_units]), name='biases', dtype=tf.float32)
    hidden1 = tf.nn.relu(tf.matmul(batch_features * 100, weights) + biases)

# Hidden 2
with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units]), dtype=tf.float32, name='weights')
    biases = tf.Variable(tf.truncated_normal([hidden2_units]), name='biases', dtype=tf.float32)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

# Linear
with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([hidden2_units, output_units]), dtype=tf.float32, name='weights')
    biases = tf.Variable(tf.truncated_normal([output_units]), name='biases', dtype=tf.float32)
    logits = tf.matmul(hidden2, weights) + biases

# Define loss and train op
batch_labels = tf.to_int64(batch_labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, batch_labels)
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

#tf.scalar_summary(loss.op.name, loss)
if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
else:
    optimizer = tf.train.MomentumOptimizer(learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
'''
batch_labels = tf.to_int64(batch_labels)
global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# Linear
weights2 = tf.Variable(tf.truncated_normal([feature_size, 2]))
biases2 = tf.Variable(tf.truncated_normal([2]))
logits2 = tf.matmul(batch_features, weights2) + biases2
cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits2, batch_labels)
loss2 = tf.reduce_mean(cross_entropy2)
train_op2 = optimizer.minimize(loss2, global_step=global_step)

validate_batch_labels = tf.to_int64(validate_batch_labels)
validate_softmax = tf.nn.softmax(tf.matmul(validate_batch_features, weights2) + biases2)

correct_prediction = tf.equal(tf.argmax(validate_softmax, 1), validate_batch_labels)
#correct_prediction = tf.equal(tf.argmax(validate_softmax,1), tf.argmax(validate_softmax,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

mode = FLAGS.mode
saver = tf.train.Saver()
steps_to_validate = FLAGS.steps_to_validate
init_op = tf.initialize_all_variables()

tf.scalar_summary('loss', loss2)
tf.scalar_summary('accuracy', accuracy)

with tf.Session() as sess:
    # Write summary for TensorBoard
    summary_op = tf.merge_all_summaries()
    output_dir = FLAGS.output_dir
    writer = tf.train.SummaryWriter(output_dir, sess.graph)

    sess.run(init_op)

    if mode == TRAIN_MODE:

        # Get coordinator and run queues to read data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            while not coord.should_stop():
                # Run train op
                _, loss_value, epoch = sess.run([train_op2, loss2, global_step])

                if epoch % steps_to_validate == 0:
                    accuracy_value, summary_value = sess.run([accuracy, summary_op])
                    print("Epoch: {}, loss: {}, accuracy: {}".format(epoch, loss_value, accuracy_value))

                    writer.add_summary(summary_value, epoch)
                    saver.save(sess, "./checkpoint/checkpoint.ckpt", global_step=epoch)

        except tf.errors.OutOfRangeError:
            print("Done training after reading all data")
        finally:
            coord.request_stop()

        # Wait for threads to exit
        coord.join(threads)

    elif mode == INFERENCE_MODE:
        print("hello inference")

        inference_data = ""
        ckpt = tf.train.get_checkpoint_state("./checkpoint/")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            w, b = sess.run([weights2, biases2])
            print("w: {}, b: {}".format(w, b))

