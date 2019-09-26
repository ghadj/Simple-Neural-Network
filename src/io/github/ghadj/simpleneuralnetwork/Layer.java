package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

public class Layer {
	private List<Neuron> neurons = new ArrayList<Neuron>();

	public void addNeuron(Neuron n){
		neurons.add(n);
	}
}
