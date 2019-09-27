package io.github.ghadj.simpleneuralnetwork;

import java.util.Random;

public class Synapse {
	private double weight;
	private double previousWeight;
	private Neuron neuronFrom;
	private Neuron NeuronTo;

	public Synapse(Neuron neuronFrom, Neuron NeuronTo) {
		this.neuronFrom = neuronFrom;
		this.NeuronTo = NeuronTo;
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

	public Neuron getNeuronFrom() {
		return neuronFrom;
	}

	public Neuron getNeuronTo() {
		return NeuronTo;
	}
}
