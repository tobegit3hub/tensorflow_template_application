package com.tobe.data

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.tensorflow.example._
import org.tensorflow.hadoop.io.TFRecordFileOutputFormat

/**
  * Generate sparse TFRecords with Spark.
  */
object GenerateSparseTfrecords {

  def main(args: Array[String]) {

    if (args.length != 2) {
      System.err.println("Should give input and output path")
      System.exit(1)
    }

    val Array(inputPath, outputPath) = args

    val sparkConf = new SparkConf().setAppName("Generate TFRecords")
    val sc = new SparkContext(sparkConf)
    val fs = FileSystem.get(sc.hadoopConfiguration)

    val features = transferToTFRecords(sc.textFile(inputPath))

    //fs.delete(new Path(outputPath), true)

    features.saveAsNewAPIHadoopFile[TFRecordFileOutputFormat](outputPath)

  }

  def transferToTFRecords(rdd: RDD[String]) = {
    rdd.map(line => {

      val arr = line.split(" ")
      val label = FloatList.newBuilder().addValue(arr(0).toFloat).build()
      val idsList = Int64List.newBuilder()
      val valuesList = FloatList.newBuilder()

      arr.filter(_.contains(":")) map (featurePair => {
        val pair = featurePair.split(":")
        val ids = pair.apply(0).toInt
        val values = pair.apply(1).toFloat
        idsList.addValue(ids)
        valuesList.addValue(values)
      })

      val features = Features.newBuilder().putFeature("label", Feature.newBuilder().setFloatList(label).build())
        .putFeature("ids", Feature.newBuilder().setInt64List(idsList.build()).build())
        .putFeature("values", Feature.newBuilder().setFloatList(valuesList.build()).build())
        .build()
      val example = Example.newBuilder()
        .setFeatures(features)
        .build()

      (new BytesWritable(example.toByteArray), NullWritable.get())
    })
  }


}
