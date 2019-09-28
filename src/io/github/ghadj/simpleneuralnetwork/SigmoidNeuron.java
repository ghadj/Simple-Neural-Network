package io.github.ghadj.simpleneuralnetwork;

public class SigmoidNeuron extends Neuron {
    private double errorSignal = 0; // δ

    private static double sigmoidFunction(double x) {
        return 1 / (1 + Math.exp(-(NeuralNetwork.SIGMOID_SLOPE * x)));
    }

    public void activate() {
        double dotProduct = 0;
        for (Synapse s : super.getSynapsesIn()) {
            dotProduct += s.getNeuronFrom().getOutput() * s.getWeight();
        }
        super.setOutput(sigmoidFunction(dotProduct));
    }

    public double getErrorSignal() {
        return errorSignal;
    }

    public void setErrorSignal(double errorSignal) {
        this.errorSignal = errorSignal;
    }
}