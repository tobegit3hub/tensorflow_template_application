# Java Predict Client

## Introduction

It is the maven project to request TensorFlow Serving in Java, Scala or Spark.

You can also use this to generate TFRecords for CSV and LIBSVM files with Spark.

## Java Predict Client

Compile the project.

```
mvn clean install -DskipTests
```

Run the client for dense model.

```
mvn exec:java -Dexec.mainClass="com.tobe.client.PredictClient" -Dexec.args="127.0.0.1 9000 dense 1"
```

Run the client for sparse model.

```
mvn compile exec:java -Dexec.mainClass="com.tobe.client.SparsePredictClient" -Dexec.args="127.0.0.1 9000 sparse 1"
```

## Scala Predict Client

Run the scala client.

```
mvn exec:java -Dexec.mainClass="com.tobe.client.ScalaDensePredictClient" -Dexec.args="127.0.0.1 9000 dense 1"
```

## Spark Predict Client

Run the client with Spark.

```
./bin/spark-submit --cluster xxx --master yarn-client --num-executors 1 --class com.tobe.client.SparkDensePredictClient --conf spark.speculation=false predict-1.0-SNAPSHOT.jar
```

## Generate TFRecords

Compile and copy the jar file to submit.

```
scp tobe@x.x.x.x:/home/tobe/code/deep_recommend_system/java_predict_client3/target/predict-1.0-SNAPSHOT.jar .
```

Generate TFRecords for LIBSVM files into HDFS.

```
$SPARK_HOME/bin/spark-submit --cluster xxx --master yarn-client --num-executors 1 --class com.tobe.data.GenerateSparseTfrecords --conf spark.speculation=false predict-1.0-SNAPSHOT.jar hdfs://foo/deep_recommend_system/data/a8a_train.libsvm hdfs:://bar/deep_recommend_system/data/a8a_train.libsvm.tfrecords
```

Generate TFRecords for LIBSVM files into FDS.

```
$SPARK_HOME/bin/spark-submit --cluster xxx --master yarn-client --num-executors 1 --class com.tobe.data.GenerateSparseTfrecords --conf spark.speculation=false predict-1.0-SNAPSHOT.jar hdfs://foo/deep_recommend_system/data/a8a_train.libsvm fds://ak:sk@mybucket.endpoint/spark_sparse_tfrecords
```

Generate TFRecords for CSV files into FDS.

```
$SPARK_HOME/bin/spark-submit --cluster xxx --master yarn-client --num-executors 1 --class com.tobe.data.GenerateDenseTfrecords --conf spark.speculation=false predict-1.0-SNAPSHOT.jar hdfs://foo/deep_recommend_system/data/a8a_train.libsvm fds://ak:sk@mybucket.endpoint/spark_sparse_tfrecords
```
