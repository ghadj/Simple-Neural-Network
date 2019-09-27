package io.github.ghadj.simpleneuralnetwork;

public class BiasNeuron extends Neuron {
  public BiasNeuron() {
    super.setOutput(-1);
  }

  @Override
  public void setOutput(double output) {
    super.setOutput(-1);
  }
}