package io.github.ghadj.simpleneuralnetwork;

/**
 * Implementation of Input unit.
 * 
 * @author Georgios Hadjiantonis
 * @since 28-09-2019
 */
public class InputNeuron extends Neuron {

  /**
   * Empty constructor. Output of input neuron is initialized to 0. 
   */
  public InputNeuron(){}

  /**
   * Constructs an input neuron and sets its output to the given value.
   * 
   * @param output
   */
  public InputNeuron(double output) {
    super.setOutput(output);
  }
}