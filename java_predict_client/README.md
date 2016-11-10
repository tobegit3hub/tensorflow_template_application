# Java Predict Client

## Introduction

It's the Java gRPC client for TensorFlow models.

## Usage

1. You need to export the TensorFlow model first.
2. Setup gRPC server with `tensorflow_model_server`.
3. Compile the Java project with `mvn clean install`.
4. Run with `mvn exec:java -Dexec.mainClass="com.tobe.PredictClient" -Dexec.args="127.0.0.1 9000 cancer 1`.

## Development

TensorFlow serving implements the general interface for inference. The intpus and outputs are `map<String, TensorProto>` which are defined in Protobuf files and we can write the gRPC client in Java to predict.

We use `mvn` and `protobuf-maven-plugin` to compile the project and generate Protobuf files. The proto files are in [./src/main/proto/](./src/main/proto/). We copy `*.proto` from [serving](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/apis) and most files from [tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow). Edit the import paths in `predict.proto` and `prediction_service.proto`. Notice that if the server upgrades the proto files, we need to do this again in client side.

Then you can compile the project or modify the code for your models easily.
