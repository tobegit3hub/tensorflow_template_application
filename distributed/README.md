## Introduction

It's the distributed version of `dense_classifier.py` which runs in distributed TensorFlow cluster.

All the weights of the model are stored in ps cluster. The ps should use CPUs only and the worker can be accelerated with GPUs which are controlled by `CUDA_VISIBLE_DEVICES`. It reads data from `TFRecords` files and use `tf.train.Supervisor` to manage checkpoint and summary files. It implements the two layer's neural network and feel free to extend to complicated models.

## Usage

If you're using GPUs, please specify `CUDA_VISIBLE_DEVICES` for ps and worker.

```
CUDA_VISIBLE_DEVICES='' python dense_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=ps --task_index=0

CUDA_VISIBLE_DEVICES='' python dense_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=ps --task_index=1

CUDA_VISIBLE_DEVICES='0' python dense_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=0

CUDA_VISIBLE_DEVICES='1' python dense_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=1
```
