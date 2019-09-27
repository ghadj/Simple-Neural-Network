package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

public class Neuron {
	private List<Synapse> synapsesIn = new ArrayList<Synapse>();
	private List<Synapse> synapsesOut = new ArrayList<Synapse>();
	private double output;

	public double getOutput() {
		return output;
	}

	public void setOutput(double output) {
		this.output = output;
	}

	public void addSynapseIn(Synapse s) {
		synapsesIn.add(s);
	}

	public List<Synapse> getSynapseIn() {
		return this.synapsesIn;
	}

	public void addSynapseOut(Synapse s) {
		synapsesOut.add(s);
	}

	public List<Synapse> getSynapseOut() {
		return this.synapsesOut;
	}
}
