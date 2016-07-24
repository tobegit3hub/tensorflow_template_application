from django.shortcuts import render

from django.http import HttpResponse

from django.views.decorators.csrf import csrf_exempt


def index(request):
    return HttpResponse("Hello, world. You should POST /cancer_predict/predict/.")



import tensorflow as tf
import numpy as np

feature_size = 9
model_path = "/Users/tobe/code/deep_recommend_system/checkpoint/"
weights2 = tf.Variable(tf.truncated_normal([feature_size, 2]))
biases2 = tf.Variable(tf.truncated_normal([2]))
inference_features = tf.placeholder("float", [None, feature_size])
inference_softmax = tf.nn.softmax(tf.matmul(inference_features, weights2) + biases2)
inference_op = tf.argmax(inference_softmax, 1)

saver = tf.train.Saver()
init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt and ckpt.model_checkpoint_path:
    print("Use the model {}".format(ckpt.model_checkpoint_path))
    # saver.restore(sess, ckpt.model_checkpoint_path)
    saver.restore(sess, "/Users/tobe/code/deep_recommend_system/checkpoint/checkpoint.cp-40")
w = sess.run(weights2)
print(w)

# Disable CSRF, refer to https://docs.djangoproject.com/en/dev/ref/csrf/#edge-cases
@csrf_exempt
def predict(request):
    if request.method == 'POST':

         inference_data = np.array([(10,10,10,8,6,1,8,9,1), (6,2,1,1,1,1,7,1,1), (2,5,3,3,6,7,7,5,1), (10,4,3,1,3,3,6,5,2), (6,10,10,2,8,10,7,3,3), (5,6,5,6,10,1,3,1,1), (1,1,1,\
1,2,1,2,1,2), (3,7,7,4,4,9,4,8,1), (1,1,1,1,2,1,2,1,1), (4,1,1,3,2,1,3,1,1) ])

         correct_labels = [1,0,1,1,1,1,0,1,0,0]
         inference_result = sess.run(inference_op, feed_dict={inference_features: inference_data})
         print("Real data is: {}".format(correct_labels))
         print("Inference data is {}".format(inference_result))
         return HttpResponse(inference_result)
    else:
         return HttpResponse("Please use POST to request with data")
