package io.github.ghadj.simpleneuralnetwork;

/**
 * Implementation of a signmoid neuron (not input/bias unit).
 * 
 * @author Georgios Hadjiantonis
 * @since 28-09-2019
 */
public class SigmoidNeuron extends Neuron {
    private double errorSignal = 0; // δ

    /**
     * Definition of the sigmoid function.
     * 
     * @param x input.
     * @return output value of the sigmoid function.
     */
    private static double sigmoidFunction(double x) {
        return 1 / (1 + Math.exp(-(NeuralNetwork.SIGMOID_SLOPE * x)));
    }

    /**
     * Activates current neuron, calculates its output based on all the incoming
     * synapses. Applies the sigmoid function on the dot product of inputs and
     * weights.
     */
    public void activate() {
        double dotProduct = 0;
        for (Synapse s : super.getSynapsesIn()) {
            dotProduct += s.getNeuronFrom().getOutput() * s.getWeight();
        }
        super.setOutput(sigmoidFunction(dotProduct));
    }

    /**
     * Returns the error signal δ.
     * 
     * @return δ.
     */
    public double getErrorSignal() {
        return errorSignal;
    }

    /**
     * Sets the error signal of the current neuron.
     * 
     * @param errorSignal new δ.
     */
    public void setErrorSignal(double errorSignal) {
        this.errorSignal = errorSignal;
    }
}