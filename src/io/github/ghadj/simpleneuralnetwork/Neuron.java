package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

/**
 * Implementation of a Neuron.
 * 
 * @author Georgios Hadjiantonis
 * @since 28-09-2019
 */
public class Neuron {
	private List<Synapse> synapsesIn = new ArrayList<Synapse>();
	private List<Synapse> synapsesOut = new ArrayList<Synapse>();
	private double output = 0.0;

	/**
	 * Returns the output of the current neuron.
	 * 
	 * @return output of the current neuron.
	 */
	public double getOutput() {
		return output;
	}

	/**
	 * Sets the output of the current neuron.
	 * 
	 * @param output value to be set.
	 */
	public void setOutput(double output) {
		this.output = output;
	}

	/**
	 * Adds a new synapse to the current neuron from another one.
	 * 
	 * @param s new synapse to current neuron.
	 */
	public void addSynapseIn(Synapse s) {
		synapsesIn.add(s);
	}

	/**
	 * Returns a list of the synapses to the current neuron.
	 * 
	 * @return list of the synapses to the current neuron.
	 */
	public List<Synapse> getSynapsesIn() {
		return this.synapsesIn;
	}

	/**
	 * Adds a new synapse from the current neuron to another one.
	 * 
	 * @param s new synapse from current neuron.
	 */
	public void addSynapseOut(Synapse s) {
		synapsesOut.add(s);
	}

	/**
	 * Returns a list of the synapses from the current neuron.
	 * 
	 * @return list of the synapses from the current neuron.
	 */
	public List<Synapse> getSynapsesOut() {
		return this.synapsesOut;
	}
}
