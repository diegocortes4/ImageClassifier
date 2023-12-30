import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ImageClassifierExample {
    private static final Logger log = LoggerFactory.getLogger(ImageClassifierExample.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64; // Batch size for training
        int numEpochs = 1; // Number of training epochs

        // Get the EMNIST dataset (replace with your dataset)
        DataSetIterator emnistTrain = new EmnistDataSetIterator(EmnistDataSetIterator.Set.TRAIN, batchSize, false);
        DataSetIterator emnistTest = new EmnistDataSetIterator(EmnistDataSetIterator.Set.TEST, batchSize, false);

        // Neural network configuration
        MultiLayerNetwork network = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(0.006, 0.9))
            .l2(1e-4)
            .list()
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nIn(1)
                .stride(1, 1)
                .nOut(20)
                .activation(Activation.IDENTITY)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(62)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .backpropType(BackpropType.Standard)
            .pretrain(false)
            .backprop(true)
            .build()
        );

        network.init();
        network.setListeners(new ScoreIterationListener(10));

        log.info("Training the model...");
        for (int i = 0; i < numEpochs; i++) {
            network.fit(emnistTrain);
        }

        log.info("Evaluating the model...");
        Evaluation evaluation = network.evaluate(emnistTest);
        log.info(evaluation.stats());

        log.info("Example completed.");
    }
}
