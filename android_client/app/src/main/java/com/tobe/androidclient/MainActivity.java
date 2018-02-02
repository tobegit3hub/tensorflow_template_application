package com.tobe.androidclient;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Arrays;


public class MainActivity extends AppCompatActivity {

    private static final String TAG = "tensorflow_client";

    static {
        // Used to load the 'native-lib' library on application startup
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        TextView tv = (TextView) findViewById(R.id.sample_text);
        tv.setText(stringFromJNI());

        // Load the TensorFlow model
        AssetManager assetManager = getAssets();
        String MODEL_FILE = "file:///android_asset/tensorflow_template_application_model.pb";
        boolean logStats = true;
        TensorFlowInferenceInterface inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);

        //final Operation operation = inferenceInterface.graphOperation(OUTPUT_NAME);
        //final int numClasses = (int) operation.output(0).shape().size(1);
        //Log.i(TAG, "Read " + 100 + " labels, output layer  " + numClasses);

        // 1. Feed th inputs
        int[] keysValues = new int[2];
        keysValues[0] = 1;
        keysValues[1] = 2;

        float[] featruesValues = new float[18];
        for (int i = 0; i < 18; i++) {
            featruesValues[i] = 1f;
        }

        String[] inputNames = new String[2];
        inputNames[0] = "keys";
        inputNames[1] = "features";

        inferenceInterface.feed(inputNames[0], keysValues, 2, 1);
        inferenceInterface.feed(inputNames[1], featruesValues, 2, 9);

        // 2. Inference the outputs
        String[] outputNames = new String[3];
        outputNames[0] = "output_keys";
        outputNames[1] = "output_prediction";
        outputNames[2] = "output_softmax";

        inferenceInterface.run(outputNames, logStats);

        // 3. Get the outputs
        int[] keysOutputs = new int[2];
        long[] predictionOutput = new long[2];
        float[] softmaxOutput = new float[4];

        inferenceInterface.fetch(outputNames[0], keysOutputs);
        inferenceInterface.fetch(outputNames[1], predictionOutput);
        inferenceInterface.fetch(outputNames[2], softmaxOutput);


        Log.v(TAG, Arrays.toString(keysOutputs));
        Log.v(TAG, Arrays.toString(predictionOutput));
        Log.v(TAG, Arrays.toString(softmaxOutput));
        tv.setText(stringFromJNI() + "\n\nKeys: " + Arrays.toString(keysOutputs) + "\nPrediction: " + Arrays.toString(predictionOutput) + "\nSoftmax: " + Arrays.toString(softmaxOutput));

    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}
