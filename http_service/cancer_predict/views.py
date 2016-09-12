from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

import tensorflow as tf
import numpy as np
import json


class PredictService(object):
    def __init__(self, checkpoint_path, checkpoint_file):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_file = checkpoint_file
        self.sess = None
        self.inputs = None
        self.outputs = None

        self.init_session_handler()

    def init_session_handler(self):
        self.sess = tf.Session()

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Use the model {}".format(ckpt.model_checkpoint_path))
            saver = tf.train.import_meta_graph(self.checkpoint_file)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

            self.inputs = json.loads(tf.get_collection('inputs')[0])
            self.outputs = json.loads(tf.get_collection('outputs')[0])
        else:
            print("No model found, exit now")
            exit()

    def predict(self, examples):
        feed_dict = {}
        for k, v in self.inputs.items():
            feed_dict[v] = np.array(examples[k])
        result = self.sess.run(self.outputs, feed_dict=feed_dict)
        print("Request examples: {}, inference result: {}".format(examples,
                                                                  result))
        return result


checkpoint_path = "../checkpoint/"
checkpoint_file = "../checkpoint/checkpoint.ckpt-10.meta"
predict_service = PredictService(checkpoint_path, checkpoint_file)


def index(request):
    return HttpResponse(
        "You should POST /cancer_predict/predict/ .")


# Disable CSRF, refer to https://docs.djangoproject.com/en/dev/ref/csrf/#edge-cases
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        # The post body should be json, such as {"key": [1.0, 2.0], "features": [[10,10,10,8,6,1,8,9,1], [6,2,1,1,1,1,7,1,1]]}
        body = json.loads(request.body)

        result = predict_service.predict(body)
        return HttpResponse("Success to predict cancer, result: {}".format(
            result))
    else:
        return HttpResponse("Please use POST to request with data")
