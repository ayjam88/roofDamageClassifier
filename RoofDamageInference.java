package ai.certifai.project.classification1;

import ai.certifai.project.classification.RoofDamageClassificationInference;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class RoofDamageInference {

    private static Logger log = LoggerFactory.getLogger(RoofDamageInference.class);

    public static void main(String[] args) throws IOException {
        int height = 80;
        int width = 80;
        int channel = 3;

        File modelSaved = new File(System.getProperty("user.dir"), "generated-models/RoofDamageClassifierCustomV2.zip");

        if(modelSaved.exists() == false){
            System.out.println("Model does not exist. Abort.");
            return;
        }

        File imageToTest = new ClassPathResource("test_samples/test_10.jpg").getFile();

        //  load the saved model
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSaved);

        //  load image for testing
        NativeImageLoader loader = new NativeImageLoader(height, width, channel);
        INDArray image = loader.asMatrix(imageToTest);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        //  pass to the neural net for prediction
        INDArray output = model.output(image);
        log.info("Label         :" + Nd4j.argMax(output, 1));
        log.info("Probabilities :" + output.toString());
    }
}
