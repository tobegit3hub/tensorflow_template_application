package com.tobe;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;

import java.io.IOException;
import java.util.logging.Logger;

import tensorflow.serving.*;
import tensorflow.serving.Predict;

/**
 * Server that manages startup/shutdown of a {@code Greeter} server.
 */
public class HelloWorldServer {
  private static final Logger logger = Logger.getLogger(HelloWorldServer.class.getName());

  /* The port on which the server should run */
  private int port = 50051;
  private Server server;

  private void start() throws IOException {
    server = ServerBuilder.forPort(port)
        .addService(new PredictionServiceImpl())
        .build()
        .start();

    logger.info("Server started, listening on " + port);
    Runtime.getRuntime().addShutdownHook(new Thread() {
      @Override
      public void run() {
        // Use stderr here since the logger may have been reset by its JVM shutdown hook.
        System.err.println("*** shutting down gRPC server since JVM is shutting down");
        HelloWorldServer.this.stop();
        System.err.println("*** server shut down");
      }
    });
  }

  private void stop() {
    if (server != null) {
      server.shutdown();
    }
  }

  /**
   * Await termination on the main thread since the grpc library uses daemon threads.
   */
  private void blockUntilShutdown() throws InterruptedException {
    if (server != null) {
      server.awaitTermination();
    }
  }

  /**
   * Main launches the server from the command line.
   */
  public static void main(String[] args) throws IOException, InterruptedException {
    final HelloWorldServer server = new HelloWorldServer();
    System.out.print("Start the grpc server");

    server.start();
    server.blockUntilShutdown();
  }

  private class PredictionServiceImpl extends PredictionServiceGrpc.PredictionServiceImplBase {

    @Override
    public void predict(Predict.PredictRequest req, StreamObserver<Predict.PredictResponse> responseObserver) {
      System.out.print("Get rpc request, inputs count: " + Integer.toString(req.getInputsCount()));
      java.util.Map<java.lang.String, org.tensorflow.framework.TensorProto> inputs = req.getInputs();

      for (java.util.Map.Entry<java.lang.String, org.tensorflow.framework.TensorProto> entry : inputs.entrySet()) {
          System.out.println("Request with the key: " + entry.getKey());
      }
      
      Predict.PredictResponse reply = Predict.PredictResponse.newBuilder().putAllOutputs(inputs).build();

      responseObserver.onNext(reply);
      responseObserver.onCompleted();
    }
  }
}
