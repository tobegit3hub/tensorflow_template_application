package com.tobe.client

object ScalaDensePredictClient {

  def main(args: Array[String]): Unit = {

    System.out.println("Start scala project")

    var host: String = "127.0.0.1"
    var port: Int = 9000
    var modelName: String = "dense"
    var modelVersion: Long = 1

    // TODO: String to int doesn't work
    // Parse command-line arguments
    if (args.length == 4) {
      host = args(0)
      //port = args(1).toInt
      modelName = args(2)
      //modelVersion = args(3).toLong
    }

    // Create dense predict client
    val client: DensePredictClient = new DensePredictClient(host, port)

    // Run predict client to send request
    client.predict_example(modelName, modelVersion)

    System.out.println("End of predict client")
    // TODO: Exit the project well
    System.exit(0)

  }

}