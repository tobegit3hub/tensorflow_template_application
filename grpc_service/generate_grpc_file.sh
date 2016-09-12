#!/bin/bash

set -x
set -e

cp predict.proto ./tensorflow/
cd ./tensorflow/
python -m grpc.tools.protoc -I./ --python_out=.. --grpc_python_out=.. ./predict.proto
rm ./predict.proto
mv ./predict_pb2.py ../
