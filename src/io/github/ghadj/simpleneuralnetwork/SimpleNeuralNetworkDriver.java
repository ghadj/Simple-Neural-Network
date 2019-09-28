package io.github.ghadj.simpleneuralnetwork;

import java.io.*;
import java.util.*;
import java.security.InvalidParameterException;

public class SimpleNeuralNetworkDriver {
	public static String[] readParameters(String filename)
			throws FileNotFoundException, IOException, InvalidParameterException {
		File file = new File(filename);
		BufferedReader br;
		String[] parameters = new String[9];
		int i = 0;

		br = new BufferedReader(new FileReader(file));
		String st;
		while ((st = br.readLine()) != null)
			parameters[i++] = st.split(" ")[1];
		br.close();

		if (i != 9)
			throw new InvalidParameterException("Invalid parameters given.");
		return parameters;
	}

	public static Map<List<Double>, List<Double>> readData(int numInputNeurons, int numOutputNeurons, String filename)
			throws FileNotFoundException, IOException, InvalidParameterException {
		Map<List<Double>, List<Double>> data = new HashMap<List<Double>, List<Double>>();
		File file = new File(filename);
		BufferedReader br;
		br = new BufferedReader(new FileReader(file));
		String st;
		while ((st = br.readLine()) != null) {
			List<Double> input = new ArrayList<>();
			List<Double> output = new ArrayList<>();
			int i = 0;
			String[] line = st.split(",");
			if (line.length != numInputNeurons + numOutputNeurons) {
				br.close();
				throw new InvalidParameterException("Inconsistent data given in file " + filename);
			}

			for (int j = 0; j < numInputNeurons; j++)
				input.add(Double.parseDouble(line[i++]));
			for (int j = 0; j < numOutputNeurons; j++)
				output.add(Double.parseDouble(line[i++]));

			data.put(input, output);
		}
		br.close();
		return data;
	}

	public static void run(String[] parameters, Map<List<Double>, List<Double>> trainingData,
			Map<List<Double>, List<Double>> testData) {
		Integer[] hiddenLayerNeurons = { Integer.parseInt(parameters[0]), Integer.parseInt(parameters[1]) };
		NeuralNetwork nn = new NeuralNetwork(Integer.parseInt(parameters[2]), Integer.parseInt(parameters[3]),
				Arrays.asList(hiddenLayerNeurons), Double.parseDouble(parameters[4]),
				Double.parseDouble(parameters[5]));
		for(int i = 0;i<Integer.parseInt(parameters[6]); i++){
			nn.run(trainingData, true);
			nn.run(testData, false);
		}
		List<Double> trainError = nn.getTrainErrorList();
	}

	public static void main(String[] args) {
		if (args.length == 0) {
			System.out.println("Error: Enter the path to the parameters.txt as an argument to the program.");
			return;
		}
		Map<List<Double>, List<Double>> trainingData, testData;
		String[] parameters;
		try {
			parameters = readParameters(args[0]);
			trainingData = readData(Integer.parseInt(parameters[2]), Integer.parseInt(parameters[3]), parameters[7]);
			testData = readData(Integer.parseInt(parameters[2]), Integer.parseInt(parameters[3]), parameters[8]);
		} catch (InvalidParameterException e) {
			System.out.println("Error: " + e.getMessage());
			return;
		} catch (FileNotFoundException e) {
			System.out.println("Error: " + e.getMessage());
			return;
		} catch (IOException e) {
			System.out.println("Error: " + e.getMessage());
			return;
		}

		run(parameters, trainingData, testData);
	}
}
