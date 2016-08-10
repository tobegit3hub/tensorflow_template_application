## Introduction

It is the general high-performance inference service which provides [gRPC](https://github.com/grpc/grpc) APIs.

If all models add the `inputs` and `outputs` in collection of the graph, inference service can read them from model file and process without knowing the detail of `inputs` or `outputs`. That means we can implement the general service for all models. If you add different tensors in the graph, you can request with different json data in the gRPC client.

```
tf.add_to_collection("inputs", json.dumps({'key': keys_placeholder.name, 'features': inference_features.name }))
tf.add_to_collection("outputs", json.dumps({'key': keys.name, 'softmax': inference_softmax.name, 'prediction': inference_op.name}))
```

## Installation

If you want to generate the protobuf files, you can install with these commands.

```
pip install -r ./requirements.txt

generate_grpc_file.sh
```

In order to serve your model, make sure that you have run the training.

```
cd ..
python cancer_classifier.py
```

## Usage

Start the gRPC server for any model.

```
python inference_server.py
```

Start the gRPC client for cancer prediction model.

```
python inference_client.py
```
