#!/bin/bash

set -x
set -e

protoc -I ./ --cpp_out=.. **/*.proto
protoc -I ./ --grpc_out=.. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` **/*.proto
