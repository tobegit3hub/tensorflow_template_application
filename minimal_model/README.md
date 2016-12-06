## Introduction

The minimal TensorFlow application for benchmarking.

## Start predict server

```
./tensorflow_model_server --port=9000 --model_name=minimal --model_base_path=./model
```

## Start predict client

```
cloudml models predict -n minial -s 127.0.0.1:9000 -f ./data.json
```
