package LDA;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.Arrays;

public class BasicLDA {
	
	static int V;
	static int D;
	static int Z;
	static int[][] typesinTopic;
	static int[] topicTotal;
	static Gibbs[] partitions;

	public static void main(String[] args) {
		
		System.out.println(java.lang.Runtime.getRuntime().maxMemory());
		System.out.print("Source file for corpus (in LDA dir, omit .txt extension): ");
		Scanner keyboard = new Scanner(System.in);
		String userInput = keyboard.nextLine();
		String inPath = "/Users/tunderwood/LDA/" + userInput + ".txt";
		String outPath = "/Users/tunderwood/LDA/" + userInput + "Model.txt";
		
		System.out.print("Number of topics? ");
		userInput = keyboard.nextLine();
		Z = Integer.parseInt(userInput);
		
		System.out.print("Number of partitions? ");
		userInput = keyboard.nextLine();
		int partitionCount = Integer.parseInt(userInput);
		
		System.out.print("Number of iterations? ");
		userInput = keyboard.nextLine();
		int iterations = Integer.parseInt(userInput);
		
		SparseTable corpus = new SparseTable(inPath, Z, partitionCount);
		
		corpus.exportWords("/Users/tunderwood/LDA/Words.txt");
		corpus.exportDocs("/Users/tunderwood/LDA/DocIDs.txt");
		
		partitions = SparseTable.partitions;
		V = SparseTable.V;
		D = SparseTable.D;
		
		typesinTopic = new int[V][Z];
		topicTotal = new int[Z];
		
		System.out.println("Begin iterations.");
		
		for (int iter = 0; iter < iterations; ++iter){
			long startTime = System.nanoTime();
			
			// We initialize the shared matrix of word-topic distributions
			// and then populate it by adding up the subsets in partitions.
			
			for (int[] row : typesinTopic) {
				Arrays.fill(row, 0);
			}
			Arrays.fill(topicTotal, 0);
			
			for (Gibbs partition : partitions) {
				int[][] subset = partition.VinT;
				int[] subtotals = partition.Zct;
				for (int j = 0; j < Z; ++ j) {
					for (int i = 0; i < V; ++i) {
						typesinTopic[i][j] += subset[i][j];
					}
					topicTotal[j] += subtotals[j];
				}
			}
			
			if (iter % 50 == 3) {
				corpus.exportTopics(outPath, topics(100));
				exportTheta("/Users/tunderwood/LDA/ThetaDistrib.txt");
			}
			
			Thread[] parallelTasks = new taskThread[partitionCount];
			for (int i = 0; i < partitionCount; ++i) {
				parallelTasks[i] = new taskThread(partitions[i], typesinTopic, topicTotal);
			}
			
			for (int i = 0; i < partitionCount; i++){
				parallelTasks[i].start();
			}
			for (int i = 0; i < partitionCount; i++){
				try {
					parallelTasks[i].join();
				} catch (InterruptedException e) {
					System.out.print(e);
				}
			}
			
			double perplexity = 0.0;
			for (Gibbs partition: partitions) {
				perplexity += partition.perplexity;
			}
			
			System.out.println("Iter " + iter + ": " + perplexity);
			long elapsedTime = System.nanoTime() - startTime;
			System.out.print("Time: " + elapsedTime + "  \n");
		}
		
		// Gibbs sampling iterations completed
		corpus.exportTopics(outPath, topics(100));
		exportTheta("/Users/tunderwood/LDA/ThetaDistrib.txt");
	}

	private static int[][] topics(int topNwords) {
		// Extracts a visualization of each topic according to the formula in
		// Blei and Lafferty, 2009. Essentially, it ranks words by this weighting:
		// probability in topic * log (prob in topic / geometric mean prob in all topics)
		// We begin by converting the integer counts to a probability matrix, phi.
		
		double[][] phi = makePhi();
		
		// Then we calculate the geometric mean of each row in phi, which
		// is a vector of per-topic probabilities for a specific word.
		
		double[] geomMean = new double[V];
		for (int w = 0; w < V; ++w) {  

			double logSum = 0.0;
			for (double prob: phi[w]) {
				logSum += Math.log(prob);
			}
			geomMean[w] = Math.exp(logSum/Z);
		}
		
		// The method is going to return a matrix listing the indices of
		// the top-ranked N words in each topic. To achieve that it produces
		// a vector of weighted scores for words in each topic and then sends
		// them to a method that simultaneously sorts scores and original indices.
		// Note that we are now proceeding through phi column-wise.
		
		int[][] topicArray = new int [Z][topNwords];
		for (int t = 0; t < Z; ++t) {
			double[] weightedTopic = new double[V];
			for (int w = 0; w < V; ++w) {
				weightedTopic[w] = phi[w][t] * Math.log(phi[w][t] / geomMean[w]);
			}
			int[] ranking = ranks(weightedTopic);
			topicArray[t] = Arrays.copyOf(ranking, topNwords);
		}
		return topicArray;
	}

	private static double[][] makePhi() {
		// turns the matrix of words-in-topics into a probability
		// matrix describing the probability that a given word will be produced
		// by a given topic
		double[][] phiArray = new double [V][Z];
		for (int t = 0; t < Z; ++t) {
			for (int w = 0; w < V; ++ w) {
				phiArray[w][t] = (typesinTopic[w][t] + 1) / (double) (topicTotal[t] + 1);     // # of this word in this topic over total # in topic
			}
		}
		return phiArray;
	}

	private static int[] ranks(double[] weightedTopic) {
		int len = weightedTopic.length;
		double temp = 0;
		int tempInt = 0;
		int[] indices = new int[len];
		for (int i = 0; i < len; ++i) {
			indices[i] = i;
		}
		boolean doMore = true;
		while (doMore == true) {
			doMore = false;
			for (int i = 0; i < len-1; ++i) {
				if (weightedTopic[i] < weightedTopic[i + 1]) {
					temp = weightedTopic[i];
					weightedTopic[i] = weightedTopic[i+1];
					weightedTopic[i+1] = temp;
					tempInt = indices[i];
					indices[i] = indices[i + 1];
					indices[i + 1] = tempInt;
					doMore = true;
				}
			}
		}
		return indices;
	}

	private static void exportTheta(String outPath) {
		int[][] Theta = new int[Z][D];
		
		for (int[] row : Theta){
			Arrays.fill(row, 0);
		}
		
		// sum global document counts from partitions
		for (Gibbs partition: partitions) {
			for (int i = 0; i < partition.d; ++i) {
				for (int j = 0; j < Z; ++j) {
					Theta[j][i + partition.startDoc] = partition.TinD[j][i];
				}
			}
		}
		
		ArrayList<String> export = new ArrayList<String>();
		for (int t = 0; t < Z; ++t) {
			String outLine = "";
			for (int doc = 0; doc < D; ++doc) {
				outLine = outLine + Theta[t][doc];
				if (doc < (D-1)) {
					outLine = outLine + ",";
				}
			}
			export.add(outLine + "\n");
		}
		LineWriter outFile = new LineWriter(outPath, false);
		String[] exportArray = export.toArray(new String[export.size()]);
		outFile.send(exportArray);
	}
	
}
