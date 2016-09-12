#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import grpc
import json
import numpy as np
from tensorflow.python.framework import tensor_util

import predict_pb2


def main():
    # Connect with the gRPC server
    server_address = "127.0.0.1:50051"
    request_timeout = 5.0
    channel = grpc.insecure_channel(server_address)
    stub = predict_pb2.PredictionServiceStub(channel)

    # Make request data
    request = predict_pb2.PredictRequest()
    samples_features = np.array(
        [[10, 10, 10, 8, 6, 1, 8, 9, 1], [10, 10, 10, 8, 6, 1, 8, 9, 1]])
    samples_keys = np.array([1, 2])
    # Convert numpy to TensorProto
    request.inputs["features"].CopyFrom(tensor_util.make_tensor_proto(
        samples_features))
    request.inputs["key"].CopyFrom(tensor_util.make_tensor_proto(samples_keys))

    # Invoke gRPC request
    response = stub.Predict(request, request_timeout)

    # Convert TensorProto to numpy
    result = {}
    for k, v in response.outputs.items():
        result[k] = tensor_util.MakeNdarray(v)
    print(result)


if __name__ == '__main__':
    main()
