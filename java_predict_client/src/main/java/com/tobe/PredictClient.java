package com.tobe;


import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyChannelBuilder;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The general predict client for TensorFlow models.
 */
public class PredictClient {
    private static final Logger logger = Logger.getLogger(PredictClient.class.getName());
    private final ManagedChannel channel;
    private final PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

    // Initialize gRPC client
    public PredictClient(String host, int port) {
        //channel = ManagedChannelBuilder.forAddress(host, port)
        channel = NettyChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext(true)
                .build();
        blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
    }

    public static void main(String[] args) {
        System.out.println("Start the predict client");

        String host = "127.0.0.1";
        int port = 9000;
        String modelName = "cancer";
        long modelVersion = 1;

        // Parse command-line arguments
        if (args.length == 4) {
            host = args[0];
            port = Integer.parseInt(args[1]);
            modelName = args[2];
            modelVersion = Long.parseLong(args[3]);
        }

        // Run predict client to send request
        PredictClient client = new PredictClient(host, port);

        try {
            client.do_predict(modelName, modelVersion);
        } catch (Exception e) {
            System.out.println(e);
        } finally {
            try {
                client.shutdown();
            } catch (Exception e) {
                System.out.println(e);
            }
        }

        System.out.println("End of predict client");
    }

    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public void do_predict(String modelName, long modelVersion) {
        // Generate keys TensorProto
        int[][] keysTensorData = new int[][]{
                {1},
                {2}
        };

        TensorProto.Builder keysTensorBuilder = TensorProto.newBuilder();

        for (int i = 0; i < keysTensorData.length; ++i) {
            for (int j = 0; j < keysTensorData[i].length; ++j) {
                keysTensorBuilder.addIntVal(keysTensorData[i][j]);
            }

        }

        TensorShapeProto.Dim keysDim1 = TensorShapeProto.Dim.newBuilder().setSize(2).build();
        TensorShapeProto.Dim keysDim2 = TensorShapeProto.Dim.newBuilder().setSize(1).build();
        TensorShapeProto keysShape = TensorShapeProto.newBuilder().addDim(keysDim1).addDim(keysDim2).build();
        keysTensorBuilder.setDtype(org.tensorflow.framework.DataType.DT_INT32).setTensorShape(keysShape);
        TensorProto keysTensorProto = keysTensorBuilder.build();

        // Generate features TensorProto
        float[][] featuresTensorData = new float[][]{
                {10f, 10f, 10f, 8f, 6f, 1f, 8f, 9f, 1f},
                {10f, 10f, 10f, 8f, 6f, 1f, 8f, 9f, 1f},
        };

        TensorProto.Builder featuresTensorBuilder = TensorProto.newBuilder();

        for (int i = 0; i < featuresTensorData.length; ++i) {
            for (int j = 0; j < featuresTensorData[i].length; ++j) {
                featuresTensorBuilder.addFloatVal(featuresTensorData[i][j]);
            }
        }

        TensorShapeProto.Dim featuresDim1 = TensorShapeProto.Dim.newBuilder().setSize(2).build();
        TensorShapeProto.Dim featuresDim2 = TensorShapeProto.Dim.newBuilder().setSize(9).build();
        TensorShapeProto featuresShape = TensorShapeProto.newBuilder().addDim(featuresDim1).addDim(featuresDim2).build();
        featuresTensorBuilder.setDtype(org.tensorflow.framework.DataType.DT_FLOAT).setTensorShape(featuresShape);
        TensorProto featuresTensorProto = featuresTensorBuilder.build();

        // Generate gRPC request
        com.google.protobuf.Int64Value version = com.google.protobuf.Int64Value.newBuilder().setValue(modelVersion).build();
        Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName(modelName).setVersion(version).build();
        Predict.PredictRequest request = Predict.PredictRequest.newBuilder().setModelSpec(modelSpec).putInputs("features", featuresTensorProto).putInputs("keys", keysTensorProto).build();

        // Request gRPC server
        Predict.PredictResponse response;
        try {
            response = blockingStub.predict(request);
            java.util.Map<java.lang.String, org.tensorflow.framework.TensorProto> outputs = response.getOutputs();
            for (java.util.Map.Entry<java.lang.String, org.tensorflow.framework.TensorProto> entry : outputs.entrySet()) {
                System.out.println("Response with the key: " + entry.getKey() + ", value: " + entry.getValue());
            }
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return;
        }
    }

}
