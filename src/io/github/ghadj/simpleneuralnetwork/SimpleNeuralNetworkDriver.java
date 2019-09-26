package io.github.ghadj.simpleneuralnetwork;

import java.util.ArrayList;

public class SimpleNeuralNetworkDriver {
	private ArrayList<Double> currentEpocheTrainError;
    private ArrayList<Double> currentEpocheTestError;
    private ArrayList<Double> allEpocheTrainError = new ArrayList<Double>();
	private ArrayList<Double> allEpocheTestError = new ArrayList<Double>();
	
	public void runEpoche(){
		this.currentEpocheTrainError = new ArrayList<Double>();
		this.currentEpocheTestError = new ArrayList<Double>();
	}
	public static void main(String[] args) { 
		
	} 
}
