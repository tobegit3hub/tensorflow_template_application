#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import grpc
import inference_pb2


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = inference_pb2.InferenceServiceStub(channel)
    response = stub.DoInference(inference_pb2.InferenceRequest(name='you'))
    print("Greeter client received: " + response.message)


if __name__ == '__main__':
    run()
