package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

public class NeuralNetwork {
    public static final double SIGMOID_SLOPE = 1.0;
    private final double MOMENTUM_FACTOR;
    private final double LEARNING_RATE;
    private List<Layer> layers = new ArrayList<Layer>();

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
        this.addLayer(inputLayer);

        for (int i = 0; i < numNeuronsPerHiddenLayer.size(); i++) {
            if (numNeuronsPerHiddenLayer.get(i) <= 0)
                continue;
            Layer l = new Layer(layers.get(layers.size() - 1));
            l.addNeuron(new BiasNeuron());
            for (int j = 0; j < numNeuronsPerHiddenLayer.get(i); j++)
                l.addNeuron(new ComputationalNeuron());
            this.addLayer(l);
        }

        Layer outputLayer = new Layer(null);
        for (int i = 0; i < numOutputNeurons; i++)
            outputLayer.addNeuron(new InputNeuron());
        this.addLayer(outputLayer);

    }

    public void addLayer(Layer l) {
        layers.add(l);
    }

    public void setInputs(ArrayList<Double> inputs) {
        int i = 0; // input index
        List<Neuron> inputNeurons = layers.get(0).getNeurons(); // assume only first layer has input units
        for (Neuron n : inputNeurons) {
            if (n instanceof InputNeuron)
                n.setOutput(inputs.get(i++));
        }
    }

    public void forwardpropagation() {
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
        for (Synapse s : n.getSynapseOut())
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

    public void backpropagation(ArrayList<Double> target) {
        // start from the last layer
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer l = layers.get(i);
            for (Neuron n : l.getNeurons()) {
                if (!(n instanceof ComputationalNeuron)) // ignore input/bias units
                    continue;
                if (i == layers.size() - 1) // output layer
                    calculateErrorSignal((ComputationalNeuron) n, target.get(i));
                else // hidden layer
                    calculateErrorSignal((ComputationalNeuron) n);

                for (Synapse s : n.getSynapseIn())
                    changeWeight(s);
            }
        }
    }

    public double getError(ArrayList<Double> target) {
        double sum = 0;
        List<Neuron> lastLayerNeurons = layers.get(layers.size() - 1).getNeurons();
        for (int i = 0; i < lastLayerNeurons.size() - 1; i++) {
            // Σ(target - actual output)^2
            sum += Math.pow(target.get(i) - lastLayerNeurons.get(i).getOutput(), 2);
        }
        return 0.5 * sum;
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

    public static void main(String args[]) {
        Integer[] n = { 2, 1, 1, 4 };
        NeuralNetwork nn = new NeuralNetwork(2, 1, Arrays.asList(n), 0.3, 0.3);
        System.out.println(nn);
        System.out.println(nn.layers.get(3).getNeurons().get(1).getSynapseIn());

    }
}