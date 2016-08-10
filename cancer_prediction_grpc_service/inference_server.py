#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent import futures
import time

import grpc

import inference_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class InferenceService(inference_pb2.InferenceServiceServicer):

  def DoInference(self, request, context):
    return inference_pb2.InferenceResponse(message='Hello, %s!' % request.name)


def serve():
  #server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  #inference_pb2.add_InferenceServiceService_to_server(InferenceService(), server)
  server = inference_pb2.beta_create_InferenceService_server(InferenceService())
  server.add_insecure_port('[::]:50051')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  serve()
