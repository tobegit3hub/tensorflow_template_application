#!/usr/bin/env python

import numpy

from grpc.beta import implementations
import tensorflow as tf

import predict_pb2
import prediction_service_pb2

tf.app.flags.DEFINE_string("host", "127.0.0.1", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 9000, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "cancer", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", 1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 10.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS


def main():
  host = FLAGS.host
  port = FLAGS.port
  model_name = FLAGS.model_name
  model_version = FLAGS.model_version
  request_timeout = FLAGS.request_timeout
  '''
  Example data:
    0 5:1 6:1 17:1 21:1 35:1 40:1 53:1 63:1 71:1 73:1 74:1 76:1 80:1 83:1
    1 5:1 7:1 17:1 22:1 36:1 40:1 51:1 63:1 67:1 73:1 74:1 76:1 81:1 83:1
  '''

  # Generate keys TensorProto
  keys = numpy.asarray([1, 2])
  keys_tensor_proto = tf.contrib.util.make_tensor_proto(keys, dtype=tf.int32)

  # Generate indexs TensorProto
  indexs = numpy.asarray([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                          [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11],
                          [0, 12], [0, 13], [1, 0], [1, 1], [1, 2], [1, 3],
                          [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
                          [1, 10], [1, 11], [1, 12], [1, 13]])
  indexs_tensor_proto = tf.contrib.util.make_tensor_proto(indexs,
                                                          dtype=tf.int64)

  # Generate ids TensorProto
  ids = numpy.asarray([5, 6, 17, 21, 35, 40, 53, 63, 71, 73, 74, 76, 80, 83, 5,
                       7, 17, 22, 36, 40, 51, 63, 67, 73, 74, 76, 81, 83])
  ids_tensor_proto = tf.contrib.util.make_tensor_proto(ids, dtype=tf.int64)

  # Generate values TensorProto
  values = numpy.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  values_tensor_proto = tf.contrib.util.make_tensor_proto(values,
                                                          dtype=tf.float32)

  # Generate values TensorProto
  shape = numpy.asarray([2, 124])
  shape_tensor_proto = tf.contrib.util.make_tensor_proto(shape, dtype=tf.int64)

  # Create gRPC client and request
  channel = implementations.insecure_channel(host, port)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  if model_version > 0:
    request.model_spec.version.value = model_version

  request.inputs["keys"].CopyFrom(keys_tensor_proto)
  request.inputs["indexs"].CopyFrom(indexs_tensor_proto)
  request.inputs["ids"].CopyFrom(ids_tensor_proto)
  request.inputs["values"].CopyFrom(values_tensor_proto)
  request.inputs["shape"].CopyFrom(shape_tensor_proto)

  # Send request
  result = stub.Predict(request, request_timeout)
  print(result)


if __name__ == '__main__':
  main()
