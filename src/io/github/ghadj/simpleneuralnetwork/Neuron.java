package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

public class Neuron {
	private List<Synapse> synapsesIn = new ArrayList<Synapse>();
	private List<Synapse> synapsesOut = new ArrayList<Synapse>();
	private static final double sigmoidSlope = 1;

	private static double sigmoidFunction(double x) {
		return 1 / (1 + Math.exp(-(sigmoidSlope * x)));
	}

	public double getOutput() {
		double dotProduct = 0;
		for (Synapse s : synapsesIn) {
			dotProduct += s.getNeuronIn().getOutput()*s.getWeight();
		}
		return sigmoidFunction(dotProduct);
	}

}
