## Introduction

It's the distributed version of `cancer_classifier.py` which runs in distributed TensorFlow cluster.

## Usage

Please specify `CUDA_VISIBLE_DEVICES` if you're running with GPUs.

```
CUDA_VISIBLE_DEVICES='' python cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=ps --task_index=0

CUDA_VISIBLE_DEVICES='' python cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=ps --task_index=1

CUDA_VISIBLE_DEVICES='0' python cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=0

CUDA_VISIBLE_DEVICES='1' python cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=1
```
