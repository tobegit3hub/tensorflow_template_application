from django.shortcuts import render

from django.http import HttpResponse

from django.views.decorators.csrf import csrf_exempt

import tensorflow as tf
import numpy as np
import json


def index(request):
    return HttpResponse(
        "Hello, world. You should POST /cancer_predict/predict/.")


def init():
  feature_size = 9
  checkpoint_path = "../../checkpoint/"
  checkpoint_path = "/home/tobe/code/deep_recommend_system/checkpoint/"
  checkpoint_file = "../../checkpoint/checkpoint.ckpt-10.meta"
  checkpoint_file = "/home/tobe/code/deep_recommend_system/checkpoint/checkpoint.ckpt-10.meta"

  sess = tf.Session()

  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  if ckpt and ckpt.model_checkpoint_path:
      print("Use the model {}".format(ckpt.model_checkpoint_path))
      saver = tf.train.import_meta_graph(checkpoint_file)
      saver.restore(sess, ckpt.model_checkpoint_path)
    
      inputs = json.loads(tf.get_collection('inputs')[0])
      outputs = json.loads(tf.get_collection('outputs')[0])
      return sess, inputs, outputs
  else:
      print("No model found, exit now")
      exit(1)


# Disable CSRF, refer to https://docs.djangoproject.com/en/dev/ref/csrf/#edge-cases
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        # cancer_features="10,10,10,8,6,1,8,9,1;6,2,1,1,1,1,7,1,1"
        body = json.loads(request.body)
        data = body.get('cancer_features')
        items = [item.split(",") for item in data.split(";")]
        inference_data = np.array(items, dtype="float")

        sess, inputs, outputs = init()

        request_examples = {"features": np.array([[10,10,10,8,6,1,8,9,1], [6,2,1,1,1,1,7,1,1]], dtype="float"), "key": np.array([1, 2], dtype="float")}

        feed_dict = {}
        for k, v in inputs.items():
            feed_dict[v] = request_examples[k]
        inference_result = sess.run(
            outputs,
            feed_dict=feed_dict)
        print("Request examples: {}, inference result: {}".format(
            request_examples, inference_result))

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
