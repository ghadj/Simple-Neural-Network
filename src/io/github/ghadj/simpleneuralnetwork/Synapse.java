package io.github.ghadj.simpleneuralnetwork;

import java.util.Random;

public class Synapse {
	private double weight;
	private double previousWeight;
	private Neuron neuronIn;
	private Neuron NeuronOut;
	
	public Synapse(Neuron neuronIn, Neuron neuronOut) {
		this.neuronIn = neuronIn;
		this.NeuronOut = neuronOut;
		this.weight = (new Random()).nextDouble() - 0.5; // initialize weight to a random number [-0.5, 0.5]
		this.previousWeight = this.weight;
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public double getPreviousWeight() {
		return previousWeight;
	}

	public void setPreviousWeight(double previousWeight) {
		this.previousWeight = previousWeight;
	}

	public Neuron getNeuronIn() {
		return neuronIn;
	}

	public Neuron getNeuronOut() {
		return NeuronOut;
	}
}
