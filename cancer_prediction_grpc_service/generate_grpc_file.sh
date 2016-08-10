#!/bin/bash

set -x
set -e

cp inference.proto ./tensorflow/
cd ./tensorflow/
python -m grpc.tools.protoc -I./ --python_out=.. --grpc_python_out=.. ./inference.proto
rm ./inference.proto
