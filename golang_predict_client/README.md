# Golang Predict Client

## Introduction

It is the predict client in golang for TensorFlow Serving.

You can compile the project easily and change the inference data for your TensorFlow models.

## Usage

Install `protoc` and requirements.

```
go get -u github.com/golang/protobuf/{proto,protoc-gen-go}
go get -u google.golang.org/grpc
```

Generate protobuf files.

```
cd ./src/

./generate_proto_files.sh
```

Compile the project.

```
# Setup $GOPATH

go build -x
```

Run the predict client.

```
# For dense model
./src --model_name dense

# For sparse model
./src --model_name sparse
```
