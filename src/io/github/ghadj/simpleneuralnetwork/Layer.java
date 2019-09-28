package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

public class Layer {
	private List<Neuron> neurons = new ArrayList<Neuron>();
	private Layer previousLayer;

	public Layer(Layer previousLayer) {
		this.previousLayer = previousLayer;
	}

	public void addNeuron(Neuron n) {
		this.neurons.add(n);
		if (this.previousLayer != null && n instanceof SigmoidNeuron)
			for (Neuron p : this.previousLayer.getNeurons()) {
				Synapse s = new Synapse(p, n);
				p.addSynapseOut(s);
				n.addSynapseIn(s);
			}
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	public int getSize() {
		return neurons.size();
	}
}
