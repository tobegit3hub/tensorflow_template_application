# Cpp Predict Server

## Introduction

TensorFlow serving is the gRPC server for general TensorFlow models.

## Installation

Follow the official documents to build with `bazel build //tensorflow_serving/model_servers:tensorflow_model_server`.

Or use the binary [tensorflow_model_server](./tensorflow_model_server/) with TensorFlow 0.11.0 in Linux.

```
Usage: tensorflow_model_server [--port=8500] [--enable_batching] [--model_name=my_name] --model_base_path=/path/to/export
```

## Usage

1. Export the TensorFlow models in local host.
2. Run server with `./tensorflow_model_server --port=9000 --model_name=cancer --model_base_path=/tmp/cancer_model`.

It is possible to run with docker container. Notice that the docker image is not public yet but easy to implement.

```
sudo docker run -d -p 9000:9000 -v /tmp/cancer_model:/tmp/cancer_model docker.d.xiaomi.net/cloud-ml/model-tensorflow-cpu:0.11.0 /model_service.py cancer /tmp/cancer_model
```
