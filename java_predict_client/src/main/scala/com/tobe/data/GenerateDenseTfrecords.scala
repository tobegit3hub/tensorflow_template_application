package com.tobe.data

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.tensorflow.example._
import org.tensorflow.hadoop.io.TFRecordFileOutputFormat

/**
  * Generate dense TFRecords with Spark.
  */
object GenerateDenseTfrecords {
  def main(args: Array[String]) {

    if (args.length != 2) {
      System.err.println("Should give input and output path")
      System.exit(1)
    }

    val Array(inputPath, outputPath) = args

    val sparkConf = new SparkConf().setAppName("TFRecord Convert")
    val sc = new SparkContext(sparkConf)
    val fs = FileSystem.get(sc.hadoopConfiguration)

    val features = transferToTFRecords(sc.textFile(inputPath))

    fs.delete(new Path(outputPath), true)
    features.saveAsNewAPIHadoopFile[TFRecordFileOutputFormat](outputPath)
  }

  def transferToTFRecords(rdd: RDD[String]) = {
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


}
