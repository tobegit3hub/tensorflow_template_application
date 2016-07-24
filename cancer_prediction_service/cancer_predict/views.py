from django.shortcuts import render

from django.http import HttpResponse

from django.views.decorators.csrf import csrf_exempt

import tensorflow as tf
import numpy as np
import json


def index(request):
    return HttpResponse(
        "Hello, world. You should POST /cancer_predict/predict/.")

# Build TensorFlow graph
init_op = tf.initialize_all_variables()
feature_size = 9
model_path = "/Users/tobe/code/deep_recommend_system/checkpoint/"

# For inference
weights2 = tf.Variable(tf.truncated_normal([feature_size, 2]))
biases2 = tf.Variable(tf.truncated_normal([2]))
inference_features = tf.placeholder("float", [None, feature_size])
inference_softmax = tf.nn.softmax(tf.matmul(inference_features, weights2) +
                                  biases2)
inference_op = tf.argmax(inference_softmax, 1)

# For online training
learning_rate = 0.01
global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
batch_labels = tf.placeholder(tf.int32, [None])
batch_features = tf.placeholder("float", [None, feature_size])
logits2 = tf.matmul(batch_features, weights2) + biases2
cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits2,
                                                                batch_labels)
loss2 = tf.reduce_mean(cross_entropy2)
train_op2 = optimizer.minimize(loss2, global_step=global_step)

sess = tf.Session()
sess.run(init_op)
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt and ckpt.model_checkpoint_path:
    print("Use the model {}".format(ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt.model_checkpoint_path)


# Disable CSRF, refer to https://docs.djangoproject.com/en/dev/ref/csrf/#edge-cases
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        # cancer_features="10,10,10,8,6,1,8,9,1;6,2,1,1,1,1,7,1,1"
        body = json.loads(request.body)
        data = body.get('cancer_features')
        items = [item.split(",") for item in data.split(";")]

        inference_data = np.array(items, dtype="float")
        inference_result = sess.run(
            inference_op,
            feed_dict={inference_features: inference_data})
        print("Inference data: {}, inference result: {}".format(
            inference_data, inference_result))

        return HttpResponse("Success to predict cancer, result: {}".format(
            inference_result))
    else:
        return HttpResponse("Please use POST to request with data")


@csrf_exempt
def online_train(request):
    if request.method == 'POST':
        # cancer_features_and_labels="10,10,10,8,6,1,8,9,1,1;6,2,1,1,1,1,7,1,1,0"
        body = json.loads(request.body)
        data = body.get('cancer_features_and_labels')
        items = data.split(";")
        feed_batch_labels = []
        for i in items:
            feed_batch_labels.append(i.split(",")[9])
        feed_batch_features = [i.split(",")[0:9] for i in items]

        _, loss_value, epoch = sess.run(
            [train_op2, loss2, global_step],
            feed_dict={batch_features: feed_batch_features,
                       batch_labels: feed_batch_labels})
        print("Epoch: {}, loss: {}".format(epoch, loss_value))
        saver.save(
            sess,
            "/Users/tobe/code/deep_recommend_system/checkpoint//checkpoint.ckpt",
            global_step=epoch)

        return HttpResponse(
            "Success to online train for cancer prediction, latest epoch: {}, loss: {}".format(
                epoch, loss_value))
    else:
        return HttpResponse("Please use POST to request with data")
