package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

public class Layer {
	private List<Neuron> neurons = new ArrayList<Neuron>();
	private Layer previousLayer;
	// private Layer nextLayer = null;

	public Layer(Layer previousLayer) {
		this.previousLayer = previousLayer;
		// if(previousLayer != null)
		// this.previousLayer.setNextLayer(this);
	}

	public void addNeuron(Neuron n){
		neurons.add(n);
		if(this.previousLayer != null && !(n instanceof InputNeuron) && !(n instanceof BiasNeuron)){
			for(Neuron p : this.previousLayer.getNeurons()){
				Synapse s = new Synapse(p, n);
				p.addSynapseOut(s);
				n.addSynapseIn(s);
			}
		}
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	public int getSize(){
		return neurons.size();
	}

	/*
	 * public Layer getNextLayer(){ return nextLayer; }
	 * 
	 * public void setNextLayer(Layer l){ this.nextLayer = l; }
	 */
}
