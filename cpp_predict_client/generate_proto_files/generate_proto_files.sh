#!/bin/bash

set -x
set -e

protoc -I ./ --grpc_out=.. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` ./*.proto
