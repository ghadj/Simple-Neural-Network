package io.github.ghadj.simpleneuralnetwork;

public class InputNeuron extends Neuron{
    public InputNeuron(double output) {
		super.setOutput(output);
    }
    
    @Override
    public void activate(){
        return;
    }
}