# Cpp Predict Client

## Introduction

TensorFlow serving is the gRPC service for general TensorFlow models. We can implement the C++ gRPC client to predict.

If you are using `bazel`, refer to to [inception_client.cc](https://github.com/tensorflow/serving/pull/300)

## Usage

Add the binary in `tensorflow_serving/example/BUILD`.

```
cc_binary(
    name = "tensorflow_model_client",
    srcs = [
        "serving_client_cpp.cc",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto",
    ],
)
```

Compile the project.

```
bazel build //tensorflow_serving/example:tensorflow_model_client
```

Run the predict client.

```
bazel-bin/tensorflow_serving/example/tensorflow_model_client
```
