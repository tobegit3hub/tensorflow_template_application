package com.tobe.client;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyChannelBuilder;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The sparse predict client for TensorFlow models and a8a data.
 */
public class SparsePredictClient {
    private static final Logger logger = Logger.getLogger(SparsePredictClient.class.getName());
    private final ManagedChannel channel;
    private final PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

    // Initialize gRPC client
    public SparsePredictClient(String host, int port) {
        //channel = ManagedChannelBuilder.forAddress(host, port)
        channel = NettyChannelBuilder.forAddress(host, port)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext(true)
                .maxMessageSize(100 * 1024 * 1024)
                .build();
        blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
    }

    public static void main(String[] args) {
        System.out.println("Start the predict client");

        String host = "127.0.0.1";
        int port = 9000;
        String modelName = "sparse";
        long modelVersion = 1;

        // Parse command-line arguments
        if (args.length == 4) {
            host = args[0];
            port = Integer.parseInt(args[1]);
            modelName = args[2];
            modelVersion = Long.parseLong(args[3]);
        }

        // Run predict client to send request
        SparsePredictClient client = new SparsePredictClient(host, port);

        try {
            client.predict_example(modelName, modelVersion);
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

    public void predict_example(String modelName, long modelVersion) {

        /*  Example data:
            0 5:1 6:1 17:1 21:1 35:1 40:1 53:1 63:1 71:1 73:1 74:1 76:1 80:1 83:1
            1 5:1 7:1 17:1 22:1 36:1 40:1 51:1 63:1 67:1 73:1 74:1 76:1 81:1 83:1
        */

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

        // Generate indexs TensorProto
        // Example: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13]]
        long[][] indexsTensorData = new long[][]{
                {0, 0},
                {0, 1},
                {0, 2},
                {0, 3},
                {0, 4},
                {0, 5},
                {0, 6},
                {0, 7},
                {0, 8},
                {0, 9},
                {0, 10},
                {0, 11},
                {0, 12},
                {0, 13},
                {1, 0},
                {1, 1},
                {1, 2},
                {1, 3},
                {1, 4},
                {1, 5},
                {1, 6},
                {1, 7},
                {1, 8},
                {1, 9},
                {1, 10},
                {1, 11},
                {1, 12},
                {1, 13}
        };

        TensorProto.Builder indexsTensorBuilder = TensorProto.newBuilder();

        for (int i = 0; i < indexsTensorData.length; ++i) {
            for (int j = 0; j < indexsTensorData[i].length; ++j) {
                indexsTensorBuilder.addInt64Val(indexsTensorData[i][j]);
            }
        }

        TensorShapeProto.Dim indexsDim1 = TensorShapeProto.Dim.newBuilder().setSize(28).build();
        TensorShapeProto.Dim indexsDim2 = TensorShapeProto.Dim.newBuilder().setSize(2).build();
        TensorShapeProto indexsShape = TensorShapeProto.newBuilder().addDim(indexsDim1).addDim(indexsDim2).build();
        indexsTensorBuilder.setDtype(DataType.DT_INT64).setTensorShape(indexsShape);
        TensorProto indexsTensorProto = indexsTensorBuilder.build();

        // Generate ids TensorProto
        // Example: [5, 6, 17, 21, 35, 40, 53, 63, 71, 73, 74, 76, 80, 83, 5, 7, 17, 22, 36, 40, 51, 63, 67, 73, 74, 76, 81, 83]
        long[] idsTensorData = new long[]{
                5, 6, 17, 21, 35, 40, 53, 63, 71, 73, 74, 76, 80, 83, 5, 7, 17, 22, 36, 40, 51, 63, 67, 73, 74, 76, 81, 83
        };

        TensorProto.Builder idsTensorBuilder = TensorProto.newBuilder();

        for (int i = 0; i < idsTensorData.length; ++i) {
            idsTensorBuilder.addInt64Val(idsTensorData[i]);
        }

        TensorShapeProto.Dim idsDim1 = TensorShapeProto.Dim.newBuilder().setSize(28).build();
        TensorShapeProto idsShape = TensorShapeProto.newBuilder().addDim(idsDim1).build();
        idsTensorBuilder.setDtype(DataType.DT_INT64).setTensorShape(idsShape);
        TensorProto idsTensorProto = idsTensorBuilder.build();

        // Generate values TensorProto
        // Example: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        float[] valuesTensorData = new float[]{
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
        };

        TensorProto.Builder valuesTensorBuilder = TensorProto.newBuilder();

        for (int i = 0; i < valuesTensorData.length; ++i) {
            valuesTensorBuilder.addFloatVal(valuesTensorData[i]);
        }

        TensorShapeProto.Dim valuesDim1 = TensorShapeProto.Dim.newBuilder().setSize(28).build();
        TensorShapeProto valuesShape = TensorShapeProto.newBuilder().addDim(valuesDim1).build();
        valuesTensorBuilder.setDtype(DataType.DT_FLOAT).setTensorShape(valuesShape);
        TensorProto valuesTensorProto = valuesTensorBuilder.build();

        // Generate shape TensorProto
        // Example: [3, 124]
        long[] shapeTensorData = new long[]{
                2, 124
        };

        TensorProto.Builder shapeTensorBuilder = TensorProto.newBuilder();

        for (int i = 0; i < shapeTensorData.length; ++i) {
            shapeTensorBuilder.addInt64Val(shapeTensorData[i]);
        }

        TensorShapeProto.Dim shapeDim1 = TensorShapeProto.Dim.newBuilder().setSize(2).build();
        TensorShapeProto shapeShape = TensorShapeProto.newBuilder().addDim(shapeDim1).build();
        shapeTensorBuilder.setDtype(DataType.DT_INT64).setTensorShape(shapeShape);
        TensorProto shapeTensorProto = shapeTensorBuilder.build();

        predict(modelName, modelVersion, keysTensorProto, indexsTensorProto, idsTensorProto, valuesTensorProto, shapeTensorProto);
    }

    public void predict(String modelName, long modelVersion, TensorProto keysTensorProto, TensorProto indexsTensorProto, TensorProto idsTensorProto, TensorProto valuesTensorProto, TensorProto shapeTensorProto) {
        // Generate gRPC request
        com.google.protobuf.Int64Value version = com.google.protobuf.Int64Value.newBuilder().setValue(modelVersion).build();
        Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName(modelName).setVersion(version).build();
        Predict.PredictRequest request = Predict.PredictRequest.newBuilder().setModelSpec(modelSpec).putInputs("keys", keysTensorProto).putInputs("indexs", indexsTensorProto).putInputs("ids", idsTensorProto).putInputs("values", valuesTensorProto).putInputs("shape", shapeTensorProto).build();

        // Request gRPC server
        Predict.PredictResponse response;
        try {
            response = blockingStub.withDeadlineAfter(10, TimeUnit.SECONDS).predict(request);
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
