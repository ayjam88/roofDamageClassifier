package ai.certifai.project.classification1;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;
import java.util.Scanner;

public class RoofDamageGUI implements ActionListener {

    private JLabel label1, label2;
    private JButton okButton, cancelButton;
    private JFrame frame;

    //  create content pane
    public RoofDamageGUI(String message1, String message2) {

        //  label1 information
        label1 = new JLabel(message1);
        label1.setSize(200, 200);
        label1.setLocation(50, -60);

        //  label2 information
        label2 = new JLabel(message2);
        label2.setSize(200, 200);
        label2.setLocation(50, -40);

        //  okButton information
        okButton = new JButton("Okay");
        okButton.setLocation(30, 80);
        okButton.setSize(90, 30);
        okButton.addActionListener(this);

        //  cancelButton information
        cancelButton = new JButton("Cancel");
        cancelButton.setLocation(140, 80);
        cancelButton.setSize(90, 30);
        cancelButton.addActionListener(this);

        // panel information
        JPanel panel = new JPanel();
        panel.setLayout(null);
        panel.add(label1);
        panel.add(label2);
        panel.add(okButton);
        panel.add(cancelButton);

        // frame information
        JFrame.setDefaultLookAndFeelDecorated(true);
        frame = new JFrame("My Roof");
        frame.add(panel, BorderLayout.CENTER);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setSize(270, 190);
        frame.setLocation(800,400);
        frame.setVisible(true);

    }

    // running main program
    public static void main(String[] args) throws IOException {

        //  run content pane class
        new RoofDamageGUI("Select an image to continue.", null);

    }

    @Override
    public void actionPerformed(ActionEvent e) {

        //  close frame
        frame.setVisible(false);

        //  run inference if okButton is clicked
        if(e.getSource() == okButton) {

            try {
                doInference();
            } catch (IOException ioException) {
                ioException.printStackTrace();
            }
        }

        //  close frame if cancelButton is clicked
        else if(e.getSource() == cancelButton) {
            frame.setVisible(false);

        }
    }

    //  inference method
    public static void doInference() throws IOException {

        //  declare select file attributes
        int response;
        JFileChooser chooser = new JFileChooser(".");
        chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        response = chooser.showOpenDialog(null);

        //  create action after selecting the file
        if(response == JFileChooser.APPROVE_OPTION) {

            //  image information
                //  28 * 28 grayscale
                //  if the image is a greyscale, implies single channel
            int height = 80;
            int width = 80;
            int channel = 3;

            //  link the saved model to the program
            File modelSaved = new File(System.getProperty("user.dir"), "generated-models/RoofDamageClassifierCustomV2.zip");

            //  get the selected file
            File imageToTest = chooser.getSelectedFile();

            //  load the saved model
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSaved);

            //  load image to test
                //  Use NativeImageLoader to convert to numerical matrix
            NativeImageLoader loader = new NativeImageLoader(height, width, channel);

            //  Get the image into an INDarray
            INDArray image = loader.asMatrix(imageToTest);

            //  Preprocessing to 0-1 or 0-255 [optional]
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(image);

            //  pass to the neural net for prediction
            INDArray output = model.output(image);

            //  naming numerical labels to word
            String[] labels = {"FLASHING", "FUNGUS", "HAIL", "WATER", "WIND"};

            //  get the numerical label result and assigned it to string
            String str = Nd4j.argMax(output, 1).toString();

            //  delete the "[" and "]" symbols and parse to integer data type
            str = str.substring(1, str.length() - 1);
            int index = Integer.parseInt(str);

            //  assign result of inference
            new RoofDamageGUI("Your roof damage is " + labels[index] + ".", "Select an image to continue.");

        }
    }
}

