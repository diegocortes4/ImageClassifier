import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ImageClassifierEnhanced {
    private static final Logger log = LoggerFactory.getLogger(ImageClassifierEnhanced.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64; // Batch size for training
        int numEpochs = 10; // Number of training epochs (increased for better training)

        // Get the EMNIST dataset (replace with your dataset)
        DataSetIterator emnistTrain = new EmnistDataSetIterator(EmnistDataSetIterator.Set.TRAIN, batchSize, false);
        DataSetIterator emnistTest = new EmnistDataSetIterator(EmnistDataSetIterator.Set.TEST, batchSize, false);

        // Define and configure the neural network architecture
        MultiLayerNetwork network = buildNetwork();

        // Initialize the network and add a listener for tracking training progress
        network.init();
        network.setListeners(new ScoreIterationListener(10));

        // Train the model
        trainModel(network, emnistTrain, numEpochs);

        // Evaluate the model
        evaluateModel(network, emnistTest);

        log.info("Example completed.");
    }

    private static MultiLayerNetwork buildNetwork() {
        // Define and configure the neural network architecture
        return new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(0.006, 0.9))
            .l2(1e-4)
            .list()
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nIn(1)
                .stride(1, 1)
                .nOut(32) // Increased the number of filters
                .activation(Activation.RELU) // Use ReLU activation
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(new DenseLayer.Builder()
                .nOut(128) // Increased the number of neurons
                .activation(Activation.RELU) // Use ReLU activation
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(62) // Adjusted the output size for your dataset
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .backpropType(BackpropType.Standard)
            .pretrain(false)
            .backprop(true)
            .build()
        );
    }

    private static void trainModel(MultiLayerNetwork network, DataSetIterator iterator, int numEpochs) {
        log.info("Training the model for {} epochs...", numEpochs);
        for (int i = 0; i < numEpochs; i++) {
            network.fit(iterator);
        }
        log.info("Training complete.");
    }

    private static void evaluateModel(MultiLayerNetwork network, DataSetIterator iterator) {
        log.info("Evaluating the model...");
        Evaluation evaluation = network.evaluate(iterator);
        log.info("Evaluation results:\n{}", evaluation.stats());
    }
}
