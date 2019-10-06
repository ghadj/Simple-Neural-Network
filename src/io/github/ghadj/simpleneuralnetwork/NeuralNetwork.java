package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

/**
 * Implementation of a neural network.
 * 
 * @author Georgios Hadjiantonis
 * @since 06-10-2019
 */
public class NeuralNetwork {
    public static final double SIGMOID_SLOPE = 1.0;
    private final double MOMENTUM_FACTOR;
    private final double LEARNING_RATE;
    private List<Layer> layers = new ArrayList<Layer>();
    private ArrayList<Double> trainErrorList = new ArrayList<Double>();
    private ArrayList<Double> trainSuccessRate = new ArrayList<Double>();
    private ArrayList<Double> testErrorList = new ArrayList<Double>();
    private ArrayList<Double> testSuccessRate = new ArrayList<Double>();

    /**
     * Constructs a neural network according to the given parameters. In case of
     * negative or zero number of input/output neurons throws
     * IllegalArgumentException exception.
     * 
     * @param numInputNeurons          number of input neurons.
     * @param numOutputNeurons         number of output neurons.
     * @param numNeuronsPerHiddenLayer list of number of neurons per hidden layer.
     * @param learningRate             learning rate.
     * @param momentumFactor           momentum factor.
     * @throws IllegalArgumentException
     */
    public NeuralNetwork(int numInputNeurons, int numOutputNeurons, List<Integer> numNeuronsPerHiddenLayer,
            double learningRate, double momentumFactor) throws IllegalArgumentException {
        this.MOMENTUM_FACTOR = momentumFactor;
        this.LEARNING_RATE = learningRate;

        if (numInputNeurons <= 0 || numOutputNeurons <= 0)
            throw new IllegalArgumentException("Number of Input and Output neurons MUST be positive");

        // build the neural network
        Layer inputLayer = new Layer(null);
        inputLayer.addNeuron(new BiasNeuron());
        for (int i = 0; i < numInputNeurons; i++)
            inputLayer.addNeuron(new InputNeuron());
        layers.add(inputLayer);

        for (int i = 0; i < numNeuronsPerHiddenLayer.size(); i++) {
            if (numNeuronsPerHiddenLayer.get(i) <= 0)
                continue;
            Layer hiddenLayer = new Layer(layers.get(layers.size() - 1));
            hiddenLayer.addNeuron(new BiasNeuron());
            for (int j = 0; j < numNeuronsPerHiddenLayer.get(i); j++)
                hiddenLayer.addNeuron(new SigmoidNeuron());
            layers.add(hiddenLayer);
        }

        Layer outputLayer = new Layer(layers.get(layers.size() - 1));
        for (int i = 0; i < numOutputNeurons; i++)
            outputLayer.addNeuron(new SigmoidNeuron());
        layers.add(outputLayer);
    }

    /**
     * Runs an NN for an epoch. In case this is a training epoch does backprop.
     * Keeps mean squared error per epoch and mean success rate per epoch in
     * separate lists.
     * 
     * @param data
     * @param train true if training epoch.
     */
    public void run(Map<List<Double>, List<Double>> data, Boolean train) {
        double sumError = 0.0;
        double sumSuccessRate = 0.0;
        for (Map.Entry<List<Double>, List<Double>> t : data.entrySet()) {
            setInputs(t.getKey());
            forwardpropagation();
            sumError += getError(t.getValue());
            sumSuccessRate += getSuccessRate(t.getValue());
            if (train)
                backpropagation(t.getValue());
        }
        if (train) {
            trainErrorList.add(sumError / data.size());
            trainSuccessRate.add(sumSuccessRate / data.size());
        } else {
            testErrorList.add(sumError / data.size());
            testSuccessRate.add(sumSuccessRate / data.size());
        }
    }

    /**
     * Sets output of the input neurons.
     * 
     * @param inputs
     */
    private void setInputs(List<Double> inputs) {
        int i = 0; // input index
        List<Neuron> inputNeurons = layers.get(0).getNeurons(); // assume only first layer has input units
        for (Neuron n : inputNeurons)
            if (n instanceof InputNeuron)
                n.setOutput(inputs.get(i++));
    }

    /**
     * Performs forward propagation by activating the neurons, starting from the
     * first layer.
     */
    private void forwardpropagation() {
        for (Layer l : layers)
            for (Neuron n : l.getNeurons())
                if (n instanceof SigmoidNeuron)
                    ((SigmoidNeuron) n).activate();
    }

    /**
     * Calculates and sets error signal(δ) for the output-layer neurons.
     * 
     * @param n      neuron.
     * @param target target output.
     */
    private void calculateErrorSignal(SigmoidNeuron n, double target) {
        double errorSignal = SIGMOID_SLOPE * n.getOutput() * (1 - n.getOutput()) * (target - n.getOutput());
        n.setErrorSignal(errorSignal);
    }

    /**
     * Calculates and sets error signal(δ) for the hidden-layer neurons.
     * 
     * @param n neuron.
     */
    private void calculateErrorSignal(SigmoidNeuron n) {
        double sum = 0;
        // Σ δpk*wp
        for (Synapse s : n.getSynapsesOut())
            sum += s.getWeight() * ((SigmoidNeuron) s.getNeuronTo()).getErrorSignal();

        double errorSignal = SIGMOID_SLOPE * n.getOutput() * (1 - n.getOutput()) * sum;
        n.setErrorSignal(errorSignal);
    }

    /**
     * Changes the weight of the given synapse according to the learning rate,
     * momentum factor and error signal specified. Sets previous weight to the
     * current one.
     * 
     * @param s synapse.
     */
    private void changeWeight(Synapse s) {
        double currentWeight = s.getWeight();
        s.setWeight(s.getWeight()
                + LEARNING_RATE * ((SigmoidNeuron) s.getNeuronTo()).getErrorSignal() * s.getNeuronFrom().getOutput()
                + MOMENTUM_FACTOR * (s.getWeight() - s.getPreviousWeight()));
        s.setPreviousWeight(currentWeight);
    }

    /**
     * Performs backpropagation using the targer data given.
     * 
     * @param target
     */
    private void backpropagation(List<Double> target) {
        // start from the last layer
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer l = layers.get(i);
            for (Neuron n : l.getNeurons()) {
                if (!(n instanceof SigmoidNeuron)) // ignore input/bias units
                    continue;
                if (i == layers.size() - 1) // output layer
                    calculateErrorSignal((SigmoidNeuron) n, target.get(l.getNeurons().indexOf(n)));
                else // hidden layer
                    calculateErrorSignal((SigmoidNeuron) n);

                for (Synapse s : n.getSynapsesIn())
                    changeWeight(s);
            }
        }
    }

    /**
     * Returns the squared error according to the target data given.
     * 
     * @param target
     * @return squared error according to the target data given.
     */
    private double getError(List<Double> target) {
        double sum = 0.0;
        List<Neuron> lastLayerNeurons = layers.get(layers.size() - 1).getNeurons();
        for (int i = 0; i < lastLayerNeurons.size(); i++)
            // Σ(target - actual output)^2
            sum += Math.pow(target.get(i) - lastLayerNeurons.get(i).getOutput(), 2);
        return 0.5 * sum;
    }

    /**
     * Returns the success rate according to the target data given.
     * 
     * @param target
     * @return success rate according to the target data given.
     */
    private double getSuccessRate(List<Double> target) {
        double success = 0;
        List<Neuron> lastLayerNeurons = layers.get(layers.size() - 1).getNeurons();
        for (int i = 0; i < lastLayerNeurons.size(); i++)
            if ((lastLayerNeurons.get(i).getOutput() >= 0.5 && target.get(i) == 1)
                    || (lastLayerNeurons.get(i).getOutput() < 0.5 && target.get(i) == 0))
                success++;
        return success / target.size();
    }

    /**
     * Returns string containg some basic info about the NN.
     * 
     * @return string containg some basic info about the NN.
     */
    @Override
    public String toString() {
        StringBuilder s = new StringBuilder("\nNeural Network Info\n");
        s.append("==========================\n");
        s.append("Layer #\tNumber of Neurons\n");
        for (Layer l : layers)
            s.append(layers.indexOf(l) + "\t" + l.getSize() + "\n");
        return s.toString();
    }

    /**
     * Returns a list containing the mean of the squared error per epoch, during
     * training.
     * 
     * @return list containing the mean of the squared error per epoch.
     */
    public ArrayList<Double> getTrainErrorList() {
        return trainErrorList;
    }

    /**
     * Returns a list containing the mean of success rate per epoch, during
     * training.
     * 
     * @return list containing the mean of the success rate per epoch.
     */
    public ArrayList<Double> getTrainSuccessRate() {
        return trainSuccessRate;
    }

    /**
     * Returns a list containing the mean of the squared error per epoch, during
     * test.
     * 
     * @return list containing the mean of the squared error per epoch.
     */
    public ArrayList<Double> getTestErrorList() {
        return testErrorList;
    }

    /**
     * Returns a list containing the mean of success rate per epoch, during test.
     * 
     * @return list containing the mean of the success rate per epoch.
     */
    public ArrayList<Double> getTestSuccessRate() {
        return testSuccessRate;
    }
}