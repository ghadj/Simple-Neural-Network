package io.github.ghadj.simpleneuralnetwork;

import java.util.Random;

/**
 * Implementation of a synapse.
 * 
 * @author Georgios Hadjiantonis
 * @since 28-09-2019
 */
public class Synapse {
	private double weight;
	private double previousWeight;
	private Neuron neuronFrom;
	private Neuron NeuronTo;

	/**
	 * Constructs a new synapse between the two neurons given. The weight of the
	 * synapse is initialized to a random number from [-0.5, 0.5].
	 * 
	 * @param neuronFrom
	 * @param NeuronTo
	 */
	public Synapse(Neuron neuronFrom, Neuron NeuronTo) {
		this.neuronFrom = neuronFrom;
		this.NeuronTo = NeuronTo;
		this.weight = (new Random()).nextDouble() - 0.5; // initialize weight
		this.previousWeight = this.weight;
	}

	/**
	 * Returns the weight of the current synapse.
	 * 
	 * @return weight of the current synapse.
	 */
	public double getWeight() {
		return weight;
	}

	/**
	 * Sets the weight of the current synapse to the given one.
	 * 
	 * @param weight
	 */
	public void setWeight(double weight) {
		this.weight = weight;
	}

	/**
	 * Returns the previous weight of the current synapse.
	 * 
	 * @return previous weight of the current synapse.
	 */
	public double getPreviousWeight() {
		return previousWeight;
	}

	/**
	 * Sets the previous weight of the current synapse to the given one.
	 * 
	 * @param previousWeight
	 */
	public void setPreviousWeight(double previousWeight) {
		this.previousWeight = previousWeight;
	}

	/**
	 * Returns the outcoming neuron.
	 * 
	 * @return outcoming neuron.
	 */
	public Neuron getNeuronFrom() {
		return neuronFrom;
	}

	/**
	 * Returns the incoming neuron.
	 * 
	 * @return incoming neuron.
	 */
	public Neuron getNeuronTo() {
		return NeuronTo;
	}
}
