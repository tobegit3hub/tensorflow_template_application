package com.tobe;


import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;

import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;


import com.google.protobuf.MapEntry;

import org.tensorflow.framework.TensorProto;

import tensorflow.serving.*;
import tensorflow.serving.Predict;

/**
 * Created by tobe on 9/12/16.
 */
public class Hello {

    //private static final Logger logger = Logger.getLogger(HelloWorldClient.class.getName());
    private static final Logger logger = Logger.getLogger(Hello.class.getName());

    private final ManagedChannel channel;
    //private final GreeterGrpc.GreeterBlockingStub blockingStub;
    private final PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

    /**
     * Construct client connecting to HelloWorld server at {@code host:port}.
     */
    public Hello(String host, int port) {
        channel = ManagedChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext(true)
                .build();
        blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
    }

    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    /**
     * Say hello to server.
     */
    public void greet() {
        logger.info("Call rpc function");
        //HelloRequest request = HelloRequest.newBuilder().setName(name).build();


        // Refer to http://stackoverflow.com/questions/39443019/how-can-i-create-tensorproto-for-tensorflow-in-java
        float[][] tensorData = new float[][]{
                {10, 10, 10, 8, 6, 1, 8, 9, 1},
                {10, 10, 10, 8, 6, 1, 8, 9, 1},
        };
        TensorProto.Builder builder = TensorProto.newBuilder();
        for (int i = 0; i < tensorData.length; ++i) {
            for (int j = 0; j < tensorData[i].length; ++j) {
                builder.addFloatVal(tensorData[i][j]);
            }
        }
        TensorProto tensorProto = builder.build();

        float[] tensorData2 = new float[]{1, 2};
        TensorProto.Builder builder2 = TensorProto.newBuilder();
        for (int i = 0; i < tensorData2.length; ++i) {
            builder2.addFloatVal(tensorData2[i]);
        }
        TensorProto tensorProto2 = builder2.build();

        // GRPC map usage, refer to https://developers.google.com/protocol-buffers/docs/reference/java-generated#map-fields

        //Predict.PredictRequest.Builder requestBuilder = Predict.PredictRequest.newBuilder();
        //Predict.PredictRequest request = requestBuilder.putWeight("intputs", tensorProto);
        //Predict.PredictRequest request = requestBuilder.putOutputs("intputs", tensorProto).build();

        //Predict.PredictRequest request = tensorflow.serving.Predict.PredictRequest.newBuilder().putIuputs("intputs", tensorProto).build();
        //Predict.PredictRequest request = tensorflow.serving.Predict.PredictRequest.newBuilder().build();
        Predict.PredictRequest request = tensorflow.serving.Predict.PredictRequest.newBuilder().putInputs("features", tensorProto).putInputs("key", tensorProto2).build();
        //Predict.PredictRequest request = tensorflow.serving.Predict.PredictRequest.newBuilder().putInputs("features", tensorProto).build();


        //tensorflow.serving.Predict.PredictRequest.Builder request_builder = tensorflow.serving.Predict.PredictRequest.newBuilder().putInputs("features", tensorProto);
        //Predict.PredictRequest request = request_builder.build();


        System.out.print("Inputs count: " + Integer.toString(request.getInputsCount()));


        Predict.PredictResponse response;
        try {
            System.out.print("Start to call rpc");
            response = blockingStub.predict(request);
            System.out.print("end of call rpc");
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return;
        }

        logger.info("End of rpc");
    }


    public static void main(String[] args) {
        System.out.println("Start the program");

        Hello client = new Hello("localhost", 50051);

        try {
            client.greet();
        } catch (Exception e) {
            //logger.log(e);
            //throw e;
        } finally {
            try {
                client.shutdown();
            } catch (Exception e) {

            }
        }


    }


}
