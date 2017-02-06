# Golang Predict Client

## Introduction

It is the predict client in golang for TensorFlow Serving.

You can compile the project easily and change the inference data for your TensorFlow models.

## Usage

Generate protobuf files.

```
cd ./src/

./generate_proto_files.sh
```

Compile the project.

```
go build -x
```

Run the predict client.

```
# For dense model
./src --model_name dense

# For sparse model
./src --model_name sparse
```
