#!/bin/bash

bazel build //tensorflow_serving/example:generic_mnist_export
bazel-bin/tensorflow_serving/example/generic_mnist_export --training_iteration=100 --export_version=1 /tmp/generic


bazel build //tensorflow_serving/example:generic_inference
bazel-bin/tensorflow_serving/example/generic_inference /tmp/generic/00000001

ps aux |grep 
netstat -nltup |grep 
