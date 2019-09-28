package io.github.ghadj.simpleneuralnetwork;

import java.io.*;
import java.util.*;
import java.security.InvalidParameterException;

/**
 * Driver of the neural network. Takes as input from the arguments the path to
 * the file, containing the parameters of the neural network. Traings the neural
 * network based on the given parameters and exports the error and success rate
 * per epoch in two separate files.
 * 
 * Assume that the parameter file has the following format:
 * mHiddenLayerOneNeurons <integer> 
 * numHiddenLayerTwoNeurons <integer>
 * numInputNeurons <integer> 
 * numOutputNeurons <integer> 
 * learningRate <double>
 * momentum <double> 
 * maxIterations <integer> 
 * trainFile <path to txt file>
 * testFile <path to txt file>
 * 
 * @author Georgios Hadjiantonis
 * @since 28-09-2019
 */
public class SimpleNeuralNetworkDriver {
	private static final String errorFilename = "errors.txt";
	private static final String successRateFilename = "successrate.txt";

	/**
	 * Reads the parameters of the neural network from the given file.
	 * 
	 * @param filename path to the file containing the parameters
	 * @return a String array containing the parameters.
	 * @throws FileNotFoundException
	 * @throws IOException
	 * @throws InvalidParameterException
	 */
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

	/**
	 * Reads data from the given file. Returns a map in the form of <input list,
	 * output list>.
	 * 
	 * @param numInputNeurons  number of input neurons.
	 * @param numOutputNeurons number of output neurons.
	 * @param filename         name of file to be read.
	 * @return a map in the form of <input list, output list>.
	 * @throws FileNotFoundException
	 * @throws IOException
	 * @throws InvalidParameterException in case the data of the given file is
	 *                                   inconsistent.
	 */
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

	/**
	 * Runs the NN based on the parameters, training and testing given data. Writes
	 * the squares error and success rate to two separate files at the end of all
	 * the iterations.
	 * 
	 * @param parameters
	 * @param trainingData
	 * @param testData
	 * @throws IOException
	 */
	public static void run(String[] parameters, Map<List<Double>, List<Double>> trainingData,
			Map<List<Double>, List<Double>> testData) throws IOException {
		Integer[] hiddenLayerNeurons = { Integer.parseInt(parameters[0]), Integer.parseInt(parameters[1]) };
		NeuralNetwork nn = new NeuralNetwork(Integer.parseInt(parameters[2]), Integer.parseInt(parameters[3]),
				Arrays.asList(hiddenLayerNeurons), Double.parseDouble(parameters[4]),
				Double.parseDouble(parameters[5]));
		for (int i = 0; i < Integer.parseInt(parameters[6]); i++) {
			nn.run(trainingData, true);
			nn.run(testData, false);
		}
		List<Double> trainError = nn.getTrainErrorList();
		List<Double> testError = nn.getTestErrorList();
		writeResults(trainError, testError, errorFilename);

		List<Double> trainSuccessRate = nn.getTrainSuccessRare();
		List<Double> testSuccessRate = nn.getTestSuccessRate();
		writeResults(trainSuccessRate, testSuccessRate, successRateFilename);
	}

	/**
	 * Writes the results in a csv format to the file given.
	 * 
	 * @param trainResults
	 * @param testResults
	 * @param filename
	 * @throws IOException
	 */
	public static void writeResults(List<Double> trainResults, List<Double> testResults, String filename)
			throws IOException {
		Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "utf-8"));
		StringBuilder str = new StringBuilder();
		for (int i = 0; i < trainResults.size() && i < testResults.size(); i++)
			str.append((i + 1) + "," + trainResults.get(i) + "," + testResults.get(i) + "\n");
		writer.write(str.toString());
		writer.close();
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

			run(parameters, trainingData, testData);
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

	}
}
