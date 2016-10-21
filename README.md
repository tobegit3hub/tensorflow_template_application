## Introduction

It's the general project to walk through the proceses of using [TensorFlow](https://github.com/tensorflow/tensorflow).

Most data is stored in CSV files and you can learn to convert them to **TFRecords**. This implements the **neural network** model which can extend to more complicated ones. It stores **checkpoints** for fault tolerance and **inference**. You can learn to use **TensorBoard** as well and the example data could be found in [cancer-deep-learning-model](https://github.com/mark-watson/cancer-deep-learning-model).

## Usage

### dense data 

The [data](./data/) format should be CSV and you can convert to TFRecords.

```
3,7,7,4,4,9,4,8,1,1
1,1,1,1,2,1,2,1,1,0
4,1,1,3,2,1,3,1,1,0
7,8,7,2,4,8,3,8,2,1
9,5,8,1,2,3,2,1,5,1
```

```
cd ./data/
python convert_cancer_to_tfrecords.py
```

### sparse data 

The [data](./data/) format should be LIBSVM and you can convert to TFRecords.

```
0 1:1 6:1 14:1 20:1 37:1 40:1 51:1 61:1 70:1 72:1 74:1 76:1 80:1 83:1
0 1:1 6:1 17:1 22:1 36:1 42:1 49:1 62:1 67:1 72:1 74:1 76:1 78:1
1 4:1 6:1 14:1 23:1 39:1 40:1 52:1 61:1 67:1 72:1 74:1 77:1 82:1 97:1
1 5:1 9:1 17:1 19:1 39:1 41:1 51:1 64:1 67:1 73:1 74:1 76:1 82:1 83:1
0 4:1 6:1 15:1 22:1 36:1 40:1 55:1 63:1 67:1 73:1 74:1 76:1 82:1 83:1
0 3:1 6:1 15:1 22:1 36:1 40:1 48:1 63:1 67:1 73:1 74:1 76:1 80:1 83:1
```

```
cd ./data/
python convert_a8a_to_tfrecords.py
```

### Develop application

On dense data, we can use the `cancer_classifier.py` to train or implement your model. Refer to [distributed](./distributed/) for distributed implementation.

```
python cancer_classifier.py
```

You can also train the model from scrath and this takes time for better auc.

```
python cancer_classifier.py --mode=train_from_scratch
```

If we want to run inference or prediction, just run with parameters.

```
python cancer_classifier.py --mode=inference
```

You can specify the GPU to train.

```
CUDA_VISIBLE_DEVICES='0'
```

All above is the same for sparse data.
```
python a8a_classifier.py [parameters]
```

### Use TensorBoard

The summary data is stored in [tensorboard](./tensorboard/) and we use TenorBoard for visualization.

```
tensorboard --logdir ./tensorboard/
```

Then go to `http://127.0.0.1:6006` in the browser.
