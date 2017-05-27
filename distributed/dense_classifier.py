#!/usr/bin/env python

import tensorflow as tf
import math
import os
import numpy as np

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
# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Hyperparameters
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


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Read TFRecords files
            filename_queue = tf.train.string_input_producer(
                tf.train.match_filenames_once("../data/cancer/cancer_train.csv.tfrecords"),
                num_epochs=epoch_number)
            label, features = read_and_decode(filename_queue)
            batch_labels, batch_features = tf.train.shuffle_batch(
                [label, features],
                batch_size=batch_size,
                num_threads=thread_number,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue)

            validate_filename_queue = tf.train.string_input_producer(
                tf.train.match_filenames_once(
                    "../data/cancer/cancer_test.csv.tfrecords"),
                num_epochs=epoch_number)
            validate_label, validate_features = read_and_decode(
                validate_filename_queue)
            validate_batch_labels, validate_batch_features = tf.train.shuffle_batch(
                [validate_label, validate_features],
                batch_size=batch_size,
                num_threads=thread_number,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue)

            # Define the model
            input_units = FEATURE_SIZE
            hidden1_units = FLAGS.hidden1
            hidden2_units = FLAGS.hidden2
            output_units = 2

            # Hidden 1
            weights1 = tf.Variable(
                tf.truncated_normal([input_units, hidden1_units]),
                dtype=tf.float32,
                name='weights')
            biases1 = tf.Variable(
                tf.truncated_normal([hidden1_units]),
                name='biases',
                dtype=tf.float32)
            hidden1 = tf.nn.relu(tf.matmul(batch_features, weights1) + biases1)

            # Hidden 2
            weights2 = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units]),
                dtype=tf.float32,
                name='weights')
            biases2 = tf.Variable(
                tf.truncated_normal([hidden2_units]),
                name='biases',
                dtype=tf.float32)
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

            # Linear
            weights3 = tf.Variable(
                tf.truncated_normal([hidden2_units, output_units]),
                dtype=tf.float32,
                name='weights')
            biases3 = tf.Variable(
                tf.truncated_normal([output_units]),
                name='biases',
                dtype=tf.float32)
            logits = tf.matmul(hidden2, weights3) + biases3

            batch_labels = tf.to_int64(batch_labels)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=batch_labels)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            if FLAGS.optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)

            # Compute accuracy
            accuracy_hidden1 = tf.nn.relu(tf.matmul(validate_batch_features,
                                                    weights1) + biases1)
            accuracy_hidden2 = tf.nn.relu(tf.matmul(accuracy_hidden1, weights2)
                                          + biases2)
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
            concated = tf.concat(axis=1, values=[indices, sparse_labels])
            outshape = tf.stack([derived_size, num_labels])
            new_validate_batch_labels = tf.sparse_to_dense(concated, outshape,
                                                           1.0, 0.0)
            _, auc_op = tf.contrib.metrics.streaming_auc(
                validate_softmax, new_validate_batch_labels)

            # Define inference op
            inference_features = tf.placeholder("float", [None, 9])
            inference_hidden1 = tf.nn.relu(tf.matmul(inference_features,
                                                     weights1) + biases1)
            inference_hidden2 = tf.nn.relu(tf.matmul(inference_hidden1,
                                                     weights2) + biases2)
            inference_logits = tf.matmul(inference_hidden2, weights3) + biases3
            inference_softmax = tf.nn.softmax(inference_logits)
            inference_op = tf.argmax(inference_softmax, 1)

            saver = tf.train.Saver()
            steps_to_validate = FLAGS.steps_to_validate
            init_op = tf.global_variables_initializer()

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('auc', auc_op)

            summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="./checkpoint/",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)

        with sv.managed_session(server.target) as sess:
            step = 0
            while not sv.should_stop() and step < 1000000:

                # Get coordinator and run queues to read data
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                try:
                    while not coord.should_stop():
                        # Run train op
                        _, loss_value, step = sess.run([train_op, loss,
                                                        global_step])

                        if step % steps_to_validate == 0:
                            accuracy_value, auc_value, summary_value = sess.run(
                                [accuracy, auc_op, summary_op])
                            print(
                                "Step: {}, loss: {}, accuracy: {}, auc: {}".format(
                                    step, loss_value, accuracy_value,
                                    auc_value))

                except tf.errors.OutOfRangeError:
                    print("Done training after reading all data")
                finally:
                    coord.request_stop()

                # Wait for threads to exit
                coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
