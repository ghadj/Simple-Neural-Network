package io.github.ghadj.simpleneuralnetwork;

/**
 * Implementation of Bias unit.
 * 
 * @author Georgios Hadjiantonis
 * @since 28-09-2019
 */
public class BiasNeuron extends Neuron {
  /**
   * Constructor sets output of the neuron to -1.
   */
  public BiasNeuron() {
    super.setOutput(-1);
  }

  /**
   * Ingores the argument given and sets output of the neuron to -1.
   * 
   * @param output
   */
  @Override
  public void setOutput(double output) {
    super.setOutput(-1);
  }
}