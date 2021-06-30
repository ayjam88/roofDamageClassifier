// This file do all data loading
package ai.certifai.project.classification1;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

// Data loading
public class RoofDamageIterator {
    private static int seed = 1234;
    private static Random rng = new Random(seed);
    private static int height = 80;
    private static int width = 80;
    private static int nChannel;
    private static double trainPerc;
    private static String[] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static int batchSizeA;
    private static int numClass;
    private static int nEpoch = 5;
    private static ParentPathLabelGenerator myLabels = new ParentPathLabelGenerator();
    private static BalancedPathFilter balancedPathFilter= new BalancedPathFilter(rng, allowedExt, myLabels);
    private static ImageTransform imgTransform;
    static InputSplit trainData, testData;

    // Build a constructor to be used in classifier
    public RoofDamageIterator() {}

    public void setup(File file, int channel, int nClass, ImageTransform imageTransform, int batchSize, double trainTestRatio) {
        nChannel = channel;
        batchSizeA = batchSize;
        numClass = nClass;
        imgTransform = imageTransform;
        trainPerc = trainTestRatio;

        FileSplit fileSplit = new FileSplit(file);

        if (trainPerc > 1) {
            throw new IllegalArgumentException("Train percentage must be lower than 100");
        }

        InputSplit[] allData = fileSplit.sample(balancedPathFilter, trainPerc, 1-trainPerc);
        trainData = allData[0];
        testData = allData[1];
    }

    private static DataSetIterator makeIterator(InputSplit split, boolean training) throws IOException {

        ImageRecordReader imgRR = new ImageRecordReader(height, width, nChannel, myLabels);

        if(training && imgTransform != null) {
            imgRR.initialize(split, imgTransform);
        } else {
            imgRR.initialize(split);
        }

        DataSetIterator iter = new RecordReaderDataSetIterator(imgRR, batchSizeA, 1, numClass);

        DataNormalization scaler = new ImagePreProcessingScaler();
        iter.setPreProcessor(scaler);

        return iter;
    }

    public DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData, true);
    }

    public DataSetIterator testIterator() throws IOException {
        return makeIterator(testData, true);
    }

}

