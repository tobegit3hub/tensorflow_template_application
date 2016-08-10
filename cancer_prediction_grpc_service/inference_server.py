#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent import futures
import time
import json
import grpc

import inference_pb2

import numpy as np
import tensorflow as tf

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class InferenceService(inference_pb2.InferenceServiceServicer):
    def __init__(self):

        #saver = tf.train.Saver()
        self.sess = tf.Session()
        # Restore wights from model file
        ckpt = tf.train.get_checkpoint_state("../checkpoint/")
        if ckpt and ckpt.model_checkpoint_path:
            print("Use the model {}".format(ckpt.model_checkpoint_path))
            saver = tf.train.import_meta_graph(
                "../checkpoint/checkpoint.ckpt-40.meta")
            saver.restore(self.sess, ckpt.model_checkpoint_path)

            self.inputs = json.loads(tf.get_collection('inputs')[0])
            self.outputs = json.loads(tf.get_collection('outputs')[0])

        else:
            print("No model found, exit")
            exit()

    def DoInference(self, request, context):
        # foo_numpy_array = np.array([(10, 10, 10, 8, 6, 1, 8, 9, 1), (6, 2, 1, 1, 1, 1, 7, 1, 1)])
        # request_example = json.dumps({"key": 1, "features": foo_numpy_array.tolist()})
        request_example = json.loads(request.data)
        request_example["features"] = np.array(request_example["features"])

        feed_dict = {}
        for key in self.inputs.keys():
            feed_dict[self.inputs[key]] = request_example[key]

        inference_result = self.sess.run(self.outputs, feed_dict=feed_dict)

        return inference_pb2.InferenceResponse(data='Hello, %s!' %
                                               inference_result)


def serve():
    #server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    #inference_pb2.add_InferenceServiceService_to_server(InferenceService(), server)
    server = inference_pb2.beta_create_InferenceService_server(
        InferenceService())
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
