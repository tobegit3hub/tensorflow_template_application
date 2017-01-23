#!/bin/bash

set -ex
# install gRPC and protoc plugin for Go, see http://www.grpc.io/docs/quickstart/go.html#generate-grpc-code
mkdir tensorflow tensorflow_serving
protoc -I generate_golang_files/ generate_golang_files/*.proto --go_out=plugins=grpc:tensorflow_serving
protoc -I generate_golang_files/ generate_golang_files/tensorflow/core/framework/* --go_out=plugins=grpc:.