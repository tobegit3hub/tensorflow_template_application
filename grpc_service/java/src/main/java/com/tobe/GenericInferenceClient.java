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
//import tensorflow.serving.Predict;
import tensorflow.serving.GenericInference;

/**
 * Created by tobe on 9/12/16.
 */
public class GenericInferenceClient {

    //private static final Logger logger = Logger.getLogger(HelloWorldClient.class.getName());
    private static final Logger logger = Logger.getLogger(GenericInferenceClient.class.getName());

    private final ManagedChannel channel;
    //private final GreeterGrpc.GreeterBlockingStub blockingStub;
    private final InferenceServiceGrpc.InferenceServiceBlockingStub blockingStub;

    /**
     * Construct client connecting to HelloWorld server at {@code host:port}.
     */
    public GenericInferenceClient(String host, int port) {
        channel = ManagedChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext(true)
                .build();
        blockingStub = InferenceServiceGrpc.newBlockingStub(channel);
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
        GenericInference.InferenceRequest request = GenericInference.InferenceRequest.newBuilder().setModel("mnist2").setVersion(1).putInputs("x", tensorProto).putInputs("x2", tensorProto).build();

        //tensorflow.serving.Predict.PredictRequest.Builder request_builder = tensorflow.serving.Predict.PredictRequest.newBuilder().putInputs("features", tensorProto);
        //Predict.PredictRequest request = request_builder.build();

        System.out.print("Inputs count: " + Integer.toString(request.getInputsCount()));

        GenericInference.InferenceResponse response;
        try {
            System.out.print("Start to call rpc");
            response = blockingStub.inference(request);
            java.util.Map<java.lang.String, org.tensorflow.framework.TensorProto> outputs = response.getOutputs();
            for (java.util.Map.Entry<java.lang.String, org.tensorflow.framework.TensorProto> entry : outputs.entrySet()) {
                System.out.println("Response with the key: " + entry.getKey());
            }

            System.out.print("end of call rpc");
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return;
        }

        logger.info("End of rpc");
    }


    public static void main(String[] args) {
        System.out.println("Start the program");

        //GenericInferenceClient client = new GenericInferenceClient("localhost", 50051);
        GenericInferenceClient client = new GenericInferenceClient("localhost", 33024);

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
