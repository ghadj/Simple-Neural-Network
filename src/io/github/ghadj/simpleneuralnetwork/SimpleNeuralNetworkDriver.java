package io.github.ghadj.simpleneuralnetwork;

import java.util.*;

public class SimpleNeuralNetworkDriver {
    private ArrayList<Double> allEpocheTrainError = new ArrayList<Double>();
	private ArrayList<Double> allEpocheTestError = new ArrayList<Double>();
	
	public void runEpoche(Map<List<Double>, List<Double>> trainData){
		double currentEpocheTrainErrorSum = 0.0;
		
		allEpocheTrainError.add(currentEpocheTrainErrorSum/trainData.size());
	}

	public void runTest(Map<List<Double>, List<Double>> testData){
		double currentEpocheTestErrorSum = 0.0;

		allEpocheTestError.add(currentEpocheTestErrorSum/testData.size());
	}
	public static void main(String[] args) { 
		
	} 
}
