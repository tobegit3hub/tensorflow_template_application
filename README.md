
# Deep Recommend System

Inspired by <https://github.com/mark-watson/cancer-deep-learning-model>.

## Installation

```
pip install -r ./requirements.txt
```

## Prepare Data

The data format should be CSV and looks like this.

```
0, 1.0, 1.0, 1.0
0, 1.5, 1.5, 1.5
0, 2.0, 2.0, 2.0
1, -1.0, -1.0, -1.0
1, -1.5, -1.5, -1.5
1, -2.0, -2.0, -2.0
```

We can convert CSV file to TFRecords and print the content.

```
cd ./data/
./convert_cancer_to_tfrecords.py
./print_cancer_tfrecords.py
```

## Develop program

We can use the `cancer_classifier.py` directly.

```
python ./cancer_classifier.py
```

And edit the model for testing.

```
# Hidden 1
with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([input_units, hidden1_units],
                            stddev=1.0 / math.sqrt(float(input_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(batch_features, weights) + biases)

# Hidden 2
with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

# Linear
with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, output_units],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')

    logits = tf.matmul(hidden2, weights) + biases
```

## Use  TensorBoard

We can write summary and use TenorBoard for visualization.

```
tensorboard --logdir ./tensorboard/
```

Then go to `http://127.0.0.1:6006`.
