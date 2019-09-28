package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

/**
 * Implementation of a layer of the Neural Network.
 * 
 * @author Georgios Hadjiantonis
 * @since 28-09-2019
 */
public class Layer {
	private List<Neuron> neurons = new ArrayList<Neuron>();
	private Layer previousLayer;

	/**
	 * Constructs a new layer and sets its previous layer to the given one.
	 * 
	 * @param previousLayer
	 */
	public Layer(Layer previousLayer) {
		this.previousLayer = previousLayer;
	}

	/**
	 * Adds the given neuron to the current layer. In case the neuron is an instance
	 * of SigmoidNeuron (not bias or input unit), sets its synapses between all the
	 * neurons of the previous layer and the given neuron.
	 * 
	 * @param n new neuron to be added.
	 */
	public void addNeuron(Neuron n) {
		this.neurons.add(n);
		if (this.previousLayer != null && n instanceof SigmoidNeuron)
			for (Neuron p : this.previousLayer.getNeurons()) {
				Synapse s = new Synapse(p, n);
				p.addSynapseOut(s);
				n.addSynapseIn(s);
			}
	}

	/**
	 * Returns the list of the neurons of the current layer.
	 * 
	 * @return list of the neurons of the current layer.
	 */
	public List<Neuron> getNeurons() {
		return neurons;
	}

	/**
	 * Returns the size, number of neurons of the current layer.
	 * 
	 * @return layer's size.
	 */
	public int getSize() {
		return neurons.size();
	}
}
