#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import grpc
import json
import numpy as np
from tensorflow.python.framework import tensor_util

import predict_pb2


def main():
    channel = grpc.insecure_channel('localhost:50051')
    stub = predict_pb2.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()

    #request.inputs["x"] = np.array([(10, 10, 10, 8, 6, 1, 8, 9, 1), (6, 2, 1, 1, 1, 1, 7, 1, 1)])
    #request.inputs["features"].CopyFrom(tensor_util.make_tensor_proto([(10, 10, 10, 8, 6, 1, 8, 9, 1)], dtype=np.float32, shape=[9]))
    #request.inputs["key"].CopyFrom(tensor_util.make_tensor_proto([(1)], dtype=np.float32, shape=[1]))

    request.inputs["features"].CopyFrom(tensor_util.make_tensor_proto(np.array([[10, 10, 10, 8, 6, 1, 8, 9, 1], [10, 10, 10, 8, 6, 1, 8, 9, 1]]), dtype=np.float32, shape=[2, 9]))
    request.inputs["key"].CopyFrom(tensor_util.make_tensor_proto(np.array([1, 2]), dtype=np.float32, shape=[2]))
    
    from tensorflow.core.framework import types_pb2

    #response = stub.Predict(request, 5.0)
    response = stub.Predict(request)

    #self.assertTrue('y' in result.outputs)
    #self.assertIs(types_pb2.DT_FLOAT, result.outputs['y'].dtype)
    #self.assertEquals(3.0, result.outputs['y'].float_val[0])

    #import ipdb;ipdb.set_trace()
    result = {}
    for k, v in response.outputs.items():
        result[k] = tensor_util.MakeNdarray(v)
    #response = tensor_util.MakeNdarray(response.outputs["key"])
    print(result)
    #response = numpy.array(res.outputs["y"].value)

if __name__ == '__main__':
    main()
