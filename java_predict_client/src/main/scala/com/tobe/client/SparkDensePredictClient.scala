package com.tobe.client

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.tensorflow.example.{Example, Feature, Features, FloatList}

object SparkDensePredictClient {

  def main(args: Array[String]): Unit = {

    System.out.println("Start spark project")

    var host: String = "127.0.0.1"
    //var host = "10.235.114.223"
    var port: Int = 9000
    var modelName: String = "dense"
    var modelVersion: Long = 1
    var inputPath = "/user/u_chendihao/deep_recommend_system/data/cancer_train.csv"
    var outputPath = "/user/u_chendihao/deep_recommend_system/predict"

    // TODO: String to int doesn't work
    // Parse command-line arguments
    if (args.length == 6) {
      host = args(0)
      //port = args(1).toInt
      modelName = args(2)
      //modelVersion = args(3).toLong
      inputPath = args(4)
      outputPath = args(5)
    }

    val sparkConf = new SparkConf().setAppName("Generate TFRecord")
    val sc = new SparkContext(sparkConf)
    val fs = FileSystem.get(sc.hadoopConfiguration)


    sc.textFile(inputPath).map(line => {
      /*
      var arr = line.split(",", 2)
      var label = arr(0).toFloat

      val client = new DensePredictClient(host, port)
      val result = client.predict_example(modelName, modelVersion)
      */

      // TODO: Generate TensorProto and request the server
      System.out.println(line)

    })

    // TODO: Change to request for each executor
    val client: DensePredictClient = new DensePredictClient(host, port)
    client.predict_example(modelName, modelVersion)
    System.out.println("End of predict client")

  }

  /*
  def csvToTFRecords(rdd: RDD[String]) = {
    rdd.map(line => {
      val arr = line.split(",", 2)
      val label = FloatList.newBuilder().addValue(arr(0).toFloat).build()

      val valuesList = FloatList.newBuilder()
      arr(1).split(",").map(value => {
        val values = value.toFloat
        valuesList.addValue(values)
      })


      val features = Features.newBuilder().putFeature("label", Feature.newBuilder().setFloatList(label).build())
        .putFeature("features", Feature.newBuilder().setFloatList(valuesList.build()).build())
        .build()
      val example = Example.newBuilder()
        .setFeatures(features)
        .build()

      (new BytesWritable(example.toByteArray), NullWritable.get())
    })
  }
  */

}