package io.github.ghadj.simpleneuralnetwork;

public class BiasNeuron extends Neuron{
    public BiasNeuron(double output) {
		super.setOutput(-1);
    }
    
    @Override
    public void activate(){
        return;
    }

    @Override
    public void setOutput(double output) {
		super.setOutput(-1);
	}
}