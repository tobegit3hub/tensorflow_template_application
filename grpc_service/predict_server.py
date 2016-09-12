#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent import futures
import time
import json
import grpc
import numpy as np
import tensorflow as tf
import logging
from tensorflow.python.framework import tensor_util

import predict_pb2

logging.basicConfig(level=logging.DEBUG)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class PredictionService(predict_pb2.PredictionServiceServicer):
    def __init__(self, checkpoint_file, graph_file):
        self.checkpoint_file = checkpoint_file
        self.graph_file = graph_file
        self.sess = None
        self.inputs = None
        self.outputs = None

        self.init_session_handler()

    def init_session_handler(self):
        self.sess = tf.Session()

        # Restore graph and weights from the model file
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_file)
        if ckpt and ckpt.model_checkpoint_path:
            logging.info("Use the model: {}".format(
                ckpt.model_checkpoint_path))
            saver = tf.train.import_meta_graph(self.graph_file)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

            self.inputs = json.loads(tf.get_collection('inputs')[0])
            self.outputs = json.loads(tf.get_collection('outputs')[0])
        else:
            logging.error("No model found, exit")
            exit()

    def Predict(self, request, context):
        """Run predict op for each request.
        
        Args:
          request: The TensorProto which contains the map of "inputs". The request.inputs looks like {'features': dtype: DT_FLOAT tensor_shape { dim { size: 2 } } tensor_content: "\000\000 A\000\000?" }.
          context: The grpc.beta._server_adaptations._FaceServicerContext object.

        Returns:
          The TensorProto which contains the map of "outputs". The response.outputs looks like {'softmax': dtype: DT_FLOAT tensor_shape { dim { size: 2 } } tensor_content: "\\\326\242=4\245k?\\\326\242=4\245k?" }
        """
        request_map = request.inputs
        feed_dict = {}
        for k, v in self.inputs.items():
            # Convert TensorProto objects to numpy
            feed_dict[v] = tensor_util.MakeNdarray(request_map[k])

        # Example result: {'key': array([ 2.,  2.], dtype=float32), 'prediction': array([1, 1]), 'softmax': array([[ 0.07951042,  0.92048955], [ 0.07951042,  0.92048955]], dtype=float32)}
        predict_result = self.sess.run(self.outputs, feed_dict=feed_dict)

        response = predict_pb2.PredictResponse()
        for k, v in predict_result.items():
            # Convert numpy objects to TensorProto
            response.outputs[k].CopyFrom(tensor_util.make_tensor_proto(v))
        return response


def serve(prediction_service):
    """Start the gRPC service."""
    logging.info("Start gRPC server with PredictionService: {}".format(vars(
        prediction_service)))

    # TODO: not able to use ThreadPoolExecutor
    #server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    #inference_pb2.add_InferenceServiceService_to_server(InferenceService(), server)
    server = predict_pb2.beta_create_PredictionService_server(
        prediction_service)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    # Specify the model files
    checkpoint_file = "../checkpoint/"
    graph_file = "../checkpoint/checkpoint.ckpt-10.meta"
    prediction_service = PredictionService(checkpoint_file, graph_file)

    serve(prediction_service)
