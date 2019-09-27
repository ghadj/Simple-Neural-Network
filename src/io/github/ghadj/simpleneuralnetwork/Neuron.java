package io.github.ghadj.simpleneuralnetwork;

import java.util.*;
/*
 * @TODO define input and bias neurons and use instance of 
 */
public class Neuron {
	private List<Synapse> synapsesIn = new ArrayList<Synapse>();
	private List<Synapse> synapsesOut = new ArrayList<Synapse>();
	private double errorSignal = 0; // δ
	private double output;

	private static double sigmoidFunction(double x) {
		return 1 / (1 + Math.exp(-(NeuralNetwork.SIGMOID_SLOPE * x)));
	}

	public void activate() {
		double dotProduct = 0;
		for (Synapse s : synapsesIn) {
			dotProduct += s.getNeuronFrom().getOutput() * s.getWeight();
		}
		output = sigmoidFunction(dotProduct);
	}

	public double getOutput() {
		return output;
	}

	public void setOutput(double output) {
		this.output = output;
	}

	public void addSynapseIn(Synapse s){
		synapsesIn.add(s);
	}

	public List<Synapse> getSynapseIn(){
		return this.synapsesIn;
	}

	public void addSynapseOut(Synapse s){
		synapsesOut.add(s);
	}

	public List<Synapse> getSynapseOut(){
		return this.synapsesOut;
	}

	public double getErrorSignal() {
		return errorSignal;
	}

	public void setErrorSignal(double errorSignal) {
		this.errorSignal = errorSignal;
	}
}
