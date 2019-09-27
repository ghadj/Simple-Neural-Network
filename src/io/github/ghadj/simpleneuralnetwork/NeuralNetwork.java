package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

public class NeuralNetwork {
    public static final double SIGMOID_SLOPE = 1.0;
    private final double MOMENTUM_FACTOR;
    private final double LEARNING_RATE;
    private List<Layer> layers = new ArrayList<Layer>();
    private ArrayList<Double> trainErrorList = new ArrayList<Double>();
    private ArrayList<Double> trainSuccessRare = new ArrayList<Double>();
    private ArrayList<Double> testErrorList = new ArrayList<Double>();
    private ArrayList<Double> testSuccessRate = new ArrayList<Double>();

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
                hiddenLayer.addNeuron(new ComputationalNeuron());
            layers.add(hiddenLayer);
        }

        Layer outputLayer = new Layer(layers.get(layers.size() - 1));
        for (int i = 0; i < numOutputNeurons; i++)
            outputLayer.addNeuron(new ComputationalNeuron());
        layers.add(outputLayer);
    }

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
            trainSuccessRare.add(sumSuccessRate / data.size());
        } else {
            testErrorList.add(sumError / data.size());
            testSuccessRate.add(sumSuccessRate / data.size());
        }
    }

    private void setInputs(List<Double> inputs) {
        int i = 0; // input index
        List<Neuron> inputNeurons = layers.get(0).getNeurons(); // assume only first layer has input units
        for (Neuron n : inputNeurons)
            if (n instanceof InputNeuron)
                n.setOutput(inputs.get(i++));
    }

    private void forwardpropagation() {
        for (Layer l : layers)
            for (Neuron n : l.getNeurons())
                if (n instanceof ComputationalNeuron)
                    ((ComputationalNeuron) n).activate();
    }

    // δ for output layer neurons
    private void calculateErrorSignal(ComputationalNeuron n, double target) {
        double errorSignal = SIGMOID_SLOPE * n.getOutput() * (1 - n.getOutput()) * (target - n.getOutput());
        n.setErrorSignal(errorSignal);
    }

    private void calculateErrorSignal(ComputationalNeuron n) {
        double sum = 0;
        // Σ δpk*wp
        for (Synapse s : n.getSynapsesOut())
            sum += s.getWeight() * ((ComputationalNeuron) s.getNeuronTo()).getErrorSignal();

        double errorSignal = SIGMOID_SLOPE * n.getOutput() * (1 - n.getOutput()) * sum;
        n.setErrorSignal(errorSignal);
    }

    private void changeWeight(Synapse s) {
        double currentWeight = s.getWeight();
        s.setWeight(
                s.getWeight()
                        + LEARNING_RATE * ((ComputationalNeuron) s.getNeuronTo()).getErrorSignal()
                                * s.getNeuronFrom().getOutput()
                        + MOMENTUM_FACTOR * (s.getWeight() - s.getPreviousWeight()));
        s.setPreviousWeight(currentWeight);
    }

    private void backpropagation(List<Double> target) {
        // start from the last layer
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer l = layers.get(i);
            for (Neuron n : l.getNeurons()) {
                if (!(n instanceof ComputationalNeuron)) // ignore input/bias units
                    continue;
                if (i == layers.size() - 1) // output layer
                    calculateErrorSignal((ComputationalNeuron) n, target.get(l.getNeurons().indexOf(n)));
                else // hidden layer
                    calculateErrorSignal((ComputationalNeuron) n);

                for (Synapse s : n.getSynapsesIn())
                    changeWeight(s);
            }
        }
    }

    private double getError(List<Double> target) {
        double sum = 0.0;
        List<Neuron> lastLayerNeurons = layers.get(layers.size() - 1).getNeurons();
        for (int i = 0; i < lastLayerNeurons.size(); i++)
            // Σ(target - actual output)^2
            sum += Math.pow(target.get(i) - lastLayerNeurons.get(i).getOutput(), 2);
        return 0.5 * sum;
    }

    private double getSuccessRate(List<Double> target) {
        double success = 0;
        List<Neuron> lastLayerNeurons = layers.get(layers.size() - 1).getNeurons();
        for (int i = 0; i < lastLayerNeurons.size(); i++)
            if ((lastLayerNeurons.get(i).getOutput() >= 0.5 && target.get(i) == 1)
                    || (lastLayerNeurons.get(i).getOutput() < 0.5 && target.get(i) == 0))
                success++;
        return success / target.size();
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder("\nNeural Network Info\n");
        s.append("==========================\n");
        s.append("Layer #\tNumber of Neurons\n");
        for (Layer l : layers)
            s.append(layers.indexOf(l) + "\t" + l.getSize() + "\n");
        return s.toString();
    }

	public ArrayList<Double> getTrainErrorList() {
		return trainErrorList;
	}

	public ArrayList<Double> getTrainSuccessRare() {
		return trainSuccessRare;
	}

	public ArrayList<Double> getTestErrorList() {
		return testErrorList;
	}

	public ArrayList<Double> getTestSuccessRate() {
		return testSuccessRate;
	}
}