package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

public class NeuralNetwork {
    public static final double SIGMOID_SLOPE = 1.0;
    private final double MOMENTUM_FACTOR;
    private final double LEARNING_RATE;
    private List<Layer> layers = new ArrayList<Layer>();

    public NeuralNetwork(double momentumFactor, double learningRate) {
        this.MOMENTUM_FACTOR = momentumFactor;
        this.LEARNING_RATE = learningRate;
    }

    public void addLayer(Layer l) {
        layers.add(l);
    }
    // @TODO
    public void setInputs(ArrayList<Double> inputs){
        // instanceof InputNeuron
    }
    // @TODO check for computational neurons 
    public void forwardpropagation() {
        for (Layer l : layers)
            for (Neuron n : l.getNeurons())
            ((ComputationalNeuron) n).activate();
    }

    // δ for output layer neurons
    private void calculateErrorSignal(Neuron n, double target) {
        double errorSignal = SIGMOID_SLOPE * n.getOutput() * (1 - n.getOutput()) * (target - n.getOutput());
        ((ComputationalNeuron) n).setErrorSignal(errorSignal);
    }

    private void calculateErrorSignal(Neuron n) {
        double sum = 0;
        // Σ δpk*wp
        for (Synapse s : n.getSynapseOut())
            sum += s.getWeight() * ((ComputationalNeuron)s.getNeuronTo()).getErrorSignal();

        double errorSignal = SIGMOID_SLOPE * n.getOutput() * (1 - n.getOutput()) * sum;
        ((ComputationalNeuron) n).setErrorSignal(errorSignal);
    }

    private void changeWeight(Synapse s) {
        s.setWeight(s.getWeight() + LEARNING_RATE * ((ComputationalNeuron)s.getNeuronTo()).getErrorSignal() * s.getNeuronFrom().getOutput()
                + MOMENTUM_FACTOR * (s.getWeight() - s.getPreviousWeight()));
    }

    public void backpropagation(ArrayList<Double> target) {
        // start from the last layer
        for (int i = layers.size(); i >= 0; i--) {
            Layer l = layers.get(i);
            for (Neuron n : l.getNeurons()) {
                if (i == layers.size()) // output layer
                    calculateErrorSignal(n, target.get(i));
                else // hidden layer
                    calculateErrorSignal(n);

                for (Synapse s : n.getSynapseIn())
                    changeWeight(s);
            }
        }
    }

    public double getError(ArrayList<Double> target) {
        double sum = 0;
        List<Neuron> lastLayerNeurons = layers.get(layers.size()).getNeurons();
        for (int i = 0; i < lastLayerNeurons.size(); i++) {
            // Σ(target - actual output)^2
            sum += Math.pow(target.get(i) - lastLayerNeurons.get(i).getOutput(), 2);
        }
        return 0.5 * sum;
    }
}