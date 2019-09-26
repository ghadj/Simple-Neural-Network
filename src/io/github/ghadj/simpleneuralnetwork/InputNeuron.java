package io.github.ghadj.simpleneuralnetwork;

public class InputNeuron extends Neuron{
    private final double output;

    public InputNeuron(double output){
        this.output = output;
    }

    @Override
    public double getOutput() {
		return output;
	}
}