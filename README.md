## Introduction

It's the general project to walk through the proceses of using [TensorFlow](https://github.com/tensorflow/tensorflow).

Most data is stored in CSV files and you can learn to convert them to **TFRecords**. This implements the **neural network** model which can extend to more complicated ones. It stores **checkpoints** for fault tolerance and **inference**. You can learn to use **TensorBoard** as well and the example data could be found in [cancer-deep-learning-model](https://github.com/mark-watson/cancer-deep-learning-model).

## Usage

### Prepare data 

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

### Develop application

We can use the `cancer_classifier.py` to train or implement your model. Refer to [distributed](./distributed/) for distributed implementation.

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

### Use TensorBoard

The summary data is stored in [tensorboard](./tensorboard/) and we use TenorBoard for visualization.

```
tensorboard --logdir ./tensorboard/
```

Then go to `http://127.0.0.1:6006` in the browser.
