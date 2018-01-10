#!/usr/bin/env python

import numpy
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string("host", "127.0.0.1", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 9000, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "default", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", -1, "TensorFlow model version")
tf.app.flags.DEFINE_string("signature_name", "", "The signature name")
tf.app.flags.DEFINE_float("request_timeout", 10.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS


def main():
  # Generate inference data
  keys = numpy.asarray([1, 2, 3])
  keys_tensor_proto = tf.contrib.util.make_tensor_proto(keys, dtype=tf.int32)
  features = numpy.asarray(
      [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 1, 1, 1, 1, 1, 1, 1],
       [9, 8, 7, 6, 5, 4, 3, 2, 1], [9, 9, 9, 9, 9, 9, 9, 9, 9]])
  features_tensor_proto = tf.contrib.util.make_tensor_proto(
      features, dtype=tf.float32)

  # Create gRPC client
  channel = implementations.insecure_channel(FLAGS.host, FLAGS.port)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  if FLAGS.model_version > 0:
    request.model_spec.version.value = FLAGS.model_version
  if FLAGS.signature_name != "":
    request.model_spec.signature_name = FLAGS.signature_name
  request.inputs["keys"].CopyFrom(keys_tensor_proto)
  request.inputs["features"].CopyFrom(features_tensor_proto)

  # Send request
  result = stub.Predict(request, FLAGS.request_timeout)
  print(result)


if __name__ == "__main__":
  main()
