#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import grpc
import inference_pb2
import json
import numpy as np


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = inference_pb2.InferenceServiceStub(channel)

    inference_data = np.array(
        [(10, 10, 10, 8, 6, 1, 8, 9, 1), (6, 2, 1, 1, 1, 1, 7, 1, 1),
         (2, 5, 3, 3, 6, 7, 7, 5, 1), (10, 4, 3, 1, 3, 3, 6, 5, 2),
         (6, 10, 10, 2, 8, 10, 7, 3, 3), (5, 6, 5, 6, 10, 1, 3, 1, 1),
         (1, 1, 1, 1, 2, 1, 2, 1, 2), (3, 7, 7, 4, 4, 9, 4, 8, 1),
         (1, 1, 1, 1, 2, 1, 2, 1, 1), (4, 1, 1, 3, 2, 1, 3, 1, 1)])
    # correct_labels = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0]

    data = json.dumps({"key": 1, "features": inference_data.tolist()})
    response = stub.DoInference(inference_pb2.InferenceRequest(data=data))

    print("Receive data: {}".format(response.data))


if __name__ == '__main__':
    run()
