# Java Predict Client

## Java gRPC Client

Request TensorFlow Serving with Java client.

```
mvn clean install -DskipTests

mvn exec:java -Dexec.mainClass="com.tobe.PredictClient" -Dexec.args="127.0.0.1 9000 cancer 1"

mvn compile exec:java -Dexec.mainClass="com.tobe.SparsePredictClient" -Dexec.args="127.0.0.1 9000 sparse 1"
```

## Generate TFRecords

Compile and copy the jar file to submit.

```
scp tobe@x.x.x.x:/home/tobe/code/deep_recommend_system/java_predict_client3/target/predict-1.0-SNAPSHOT.jar .
```

Read raw file and generate TFRecords into HDFS.

```
$SPARK_HOME/bin/spark-submit --cluster xxx --master yarn-client --num-executors 1 --class com.tobe.data.GenerateSparseTfrecords --conf spark.speculation=false predict-1.0-SNAPSHOT.jar hdfs://foo/deep_recommend_system/data/a8a_train.libsvm hdfs:://bar/deep_recommend_system/data/a8a_train.libsvm.tfrecords
```

Read raw file and generate TFRecords into FDS.

```
$SPARK_HOME/bin/spark-submit --cluster xxx --master yarn-client --num-executors 1 --class com.tobe.data.GenerateSparseTfrecords --conf spark.speculation=false predict-1.0-SNAPSHOT.jar hdfs://foo/deep_recommend_system/data/a8a_train.libsvm fds://ak:sk@mybucket.endpoint/spark_sparse_tfrecords
```
