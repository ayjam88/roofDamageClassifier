// This file run the actual program and model configuration
package ai.certifai.project.classification1;

import ai.certifai.solution.classification.transferlearning.EditAtBottleneckAndExtendModel;
import ai.certifai.training.classification.GenderIterator;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class RoofDamageClassifier {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(RoofDamageClassifier.class);
    private static int seed = 122;
    private static int nChannels = 3;
    private static double lr = 1e-3;
    private static int nEpoch = 100;
    private static int batchSize = 13;
    private static int nOutput = 5;
    private static int height = 80;
    private static int width = 80;


    public static void main(String[] args) throws IOException {

        File myFile = new ClassPathResource("RoofDamage").getFile();

        ImageTransform HFlip = new FlipImageTransform(1);
        ImageTransform rCrop = new RandomCropTransform(seed, 50, 50);
        ImageTransform rotate = new RotateImageTransform(5);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(HFlip, 0.3),
                new Pair<>(rCrop, 0.3),
                new Pair<>(rotate, 0.3)
//                new Pair<>(show, 0.1)
        );

        ImageTransform tp = new PipelineImageTransform(pipeline, false);

        RoofDamageIterator roofDamageIterator = new RoofDamageIterator();

        roofDamageIterator.setup(myFile, nChannels, nOutput, tp, batchSize, 0.9);

        DataSetIterator trainIter = roofDamageIterator.trainIterator();
        DataSetIterator testIter = roofDamageIterator.testIterator();


        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nIn(nChannels)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(512)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(120)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(60)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(30)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(nOutput)
                        .build())
                .setInputType(InputType.convolutional(height, width, nChannels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        System.out.println(model.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        uiServer.attach(storage);

        model.setListeners(new ScoreIterationListener(10), new StatsListener(storage, 10));
        model.fit(trainIter, nEpoch);

        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);

        System.out.println("Train: " + evalTrain.stats());
        System.out.println("Test: " + evalTest.stats());

        //  save model
//        File locationToSave = new File(System.getProperty("user.dir"), "generated-models/RoofDamageClassifierCustomTakeChart.zip");
//
//        ModelSerializer.writeModel(model, locationToSave, false);
    }
}





