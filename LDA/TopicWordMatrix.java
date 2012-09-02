package LDA;

import java.util.Arrays;

public class TopicWordMatrix {
	// This class is a synchronized wrapper for data objects that need to be shared by threads
	// working on different partitions of the corpus -- namely a matrix storing the number of
	// words of a given type in each topic, and an array storing the total number of words
	// in each of the topics.
	//
	// All methods are synchronized.
	
	private static int[][] topicwordMatrix;
	private static int[] topicTotals;
	private static int numberofWords;
	private static int numberofTopics;
	
	public TopicWordMatrix() {
		
	}
	
	public TopicWordMatrix(int words, int topics) {
		topicwordMatrix = new int[words][topics];
		topicTotals = new int[topics];
		numberofWords = words;
		numberofTopics = topics;
	}
	
	public synchronized void initializeMatrix() {
		for (int[] row: topicwordMatrix) {
			Arrays.fill(row, 0);
		}
		Arrays.fill(topicTotals, 0);
	}
	
	public synchronized void addtoMatrix(int[][] addition, int[] newTotals) {
		int firstdimension = addition.length;
		int seconddimension = addition[0].length;
		if (firstdimension != numberofWords | seconddimension != numberofTopics) {
			System.out.println("Error: incompatible array dimensions in the topic-words matrix.");
			System.out.println("Trying to add " + firstdimension + " by " + seconddimension);
			System.out.println("to " + numberofWords + " by " + numberofTopics);
		}
		
		for (int i = 0; i < firstdimension; ++ i) {
			for (int j = 0; j < seconddimension; ++ j) {
				topicwordMatrix[i][j] += addition[i][j];
			}
		}
		for (int j = 0; j < seconddimension; ++ j) {
			topicTotals[j] += newTotals[j];
		}
	}
	
	public synchronized int[][] getMatrix() {
		return topicwordMatrix;
	}
	
	public synchronized int[] getTotals() {
		return topicTotals;
	}

}
