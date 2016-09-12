#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent import futures
import time
import json
import grpc
import numpy as np
import tensorflow as tf
import logging

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
            #import ipdb;ipdb.set_trace()

        else:
            logging.error("No model found, exit")
            exit()

    def Predict(self, request, context):

        request_map = request.inputs

        from tensorflow.python.framework import tensor_util

        feed_dict = {}
        for key in self.inputs.keys():
            
            feed_dict[self.inputs[key]] = tensor_util.MakeNdarray(request_map[key])

        

        feed_dict2 = {}
        feed_dict2['Placeholder:0'] = np.array([[10, 10, 10, 8, 6, 1, 8, 9, 1], [10, 10, 10, 8, 6, 1, 8, 9, 1]])
        feed_dict2['Placeholder_1:0'] = np.array([1, 2])
        predict_result = self.sess.run(self.outputs, feed_dict=feed_dict)
        #import ipdb;ipdb.set_trace()

        #{u'key': array([ 2.,  2.], dtype=float32),
        #u'prediction': array([1, 1]),
        #u'softmax': array([[ 0.07951042,  0.92048955],
        #[ 0.07951042,  0.92048955]], dtype=float32)}

        response = predict_pb2.PredictResponse()
        for k, v in predict_result.items():
            response.outputs[k].CopyFrom(tensor_util.make_tensor_proto(v))
        return response
        #response.outputs["key"].CopyFrom(tensor_util.make_tensor_proto(np.array([1, 2]), dtype=np.float32, shape=[2]))


        '''
        # foo_numpy_array = np.array([(10, 10, 10, 8, 6, 1, 8, 9, 1), (6, 2, 1, 1, 1, 1, 7, 1, 1)])
        # request_example = json.dumps({"key": 1, "features": foo_numpy_array.tolist()})
        request_example = json.loads(request.data)
        # TODO: support data with shape
        # request_example["features"] = np.array(request_example["features"])

        feed_dict = {}
        for key in self.inputs.keys():
            feed_dict[self.inputs[key]] = request_example[key]

        
        inference_result = self.sess.run(self.outputs, feed_dict=feed_dict)
        response_data = str(inference_result)

        logging.debug("Request data: {}, response data: {}".format(
            request.data, response_data))
        return inference_pb2.InferenceResponse(data=response_data)
        '''


def serve(prediction_service):
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
    checkpoint_file = "../checkpoint/"
    graph_file = "../checkpoint/checkpoint.ckpt-10.meta"
    prediction_service = PredictionService(checkpoint_file, graph_file)

    serve(prediction_service)
