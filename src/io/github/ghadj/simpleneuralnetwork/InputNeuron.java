package io.github.ghadj.simpleneuralnetwork;

public class InputNeuron extends Neuron {
  public InputNeuron() {
    super.setOutput(0);
  }

  public InputNeuron(double output) {
    super.setOutput(output);
  }
}