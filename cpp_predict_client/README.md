# Python Predict Client

## Introduction

TensorFlow serving is the gRPC service for general TensorFlow models. We can implement the C++ gRPC client to predict.

## Usage

1. Export the TensorFlow models.
2. Run with `./predict_client.py --host 127.0.0.1 --port 9000 --model_name cancer --model_version 1`

## Development

Build protobuf from source and refer to <https://github.com/google/protobuf/blob/master/src/README.md>.

Generetate code from proto files.
