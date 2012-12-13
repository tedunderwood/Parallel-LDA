package LDA;
import java.util.ArrayList;
import static java.util.Arrays.fill;
import java.util.Random;

public class SparseTable {
	ArrayList<String> wordTypes = new ArrayList<String>();
	ArrayList<String> docTypes = new ArrayList<String>();
	static int D = 0;
	// number of documents in table
	static int V = 0;
	// number of words in vocabulary
	static int N = 0;
	// number of tokens in corpus
	static int[] wordCounts;
	static int minTime = 3000;
	static int maxTime = 0;
	static Random randomize = new Random();
	static Gibbs[] partitions;

	public SparseTable(String filePath, int numTopics, int partitionCount) {
		// The input file is a sparse table where each line contains
		// a document ID, a word, a number of occurrences of that word, and a date/time.
		
		LineReader inFile = new LineReader(filePath);
		String[] fileLines = inFile.readlines();
		int position = 0;
		int wordCount = 0;
		ArrayList<Integer> docSizes = new ArrayList<Integer>();

		// we start by scanning the table to establish vocabulary, doc list, and corpus size
		for (String line : fileLines) {
			String[] tokens = line.split("[\t]");
			int tokenCount = tokens.length;
			if (tokenCount != 4) {
				System.out.println("Error: tokenCount not equal to 4 at " + line);
				continue;
			}
			String doc = tokens[0];
			String word = tokens[1];
			String occurs = tokens[2];
			String docTime = tokens[3];
			position = wordTypes.indexOf(word);
			if (position < 0) {
				wordTypes.add(word);
			}
			position = docTypes.indexOf(doc);
			if (position < 0) {
				docTypes.add(doc);
				docSizes.add(0);
				position = docTypes.indexOf(doc);
			}
			wordCount = Integer.parseInt(occurs);
			N += wordCount;
			docSizes.set(position, docSizes.get(position) + wordCount);
			int time = Integer.parseInt(docTime);
			if (time < minTime) {
				minTime = time;
			}
			if (time > maxTime) {
				maxTime = time;
			}
		}
		// The partition strategy adopted here assumes that the input file is already sorted by
		// document. I can arrange to make that true. Basically, we partition the corpus into 
		// runs of documents by summing the words in each document until we get close to or
		// over a number of words that would represent an equal division of the corpus.
		
		D = docTypes.size();
		V = wordTypes.size();
		System.out.println("D " + D + " V " + V + " N " + N + " Z " + numTopics + " min " + minTime + " max " + maxTime);
		
		int halfAverageDocsize = (N / D) / 2;
		int wordsPerPart = N / partitionCount;
		int wordsSofar = 0;
		int[] partitionCuts = new int[partitionCount];
		int[] partitionSizes = new int[partitionCount];
		fill(partitionCuts, 0);
		int partitionCounter = 0;
		
		for (int i=0; i < D; ++i){
			wordsSofar += docSizes.get(i);
			if (wordsSofar + halfAverageDocsize > wordsPerPart | i == D-1) {
				partitionCuts[partitionCounter] = i;
				partitionSizes[partitionCounter] = wordsSofar;
				wordsSofar = 0;
				++ partitionCounter;
			}
		}
		partitionCuts[partitionCount - 1] = D - 1;
		
		for (int i = 0; i < partitionCount; ++i){
			System.out.println(partitionCuts[i] + " - " + partitionSizes[i]);
		}
		
		// Now actually build partitions.
		partitions = new Gibbs[partitionCount];
		
		int filePosition = 0;
		
		wordCounts = new int[V];
		fill(wordCounts, 0);
		
		for (int part = 0; part < partitionCount; ++part){
			int[] docIDs = new int[partitionSizes[part]];
			int[] wordIDs = new int[partitionSizes[part]];
			int[] timestamps = new int[partitionSizes[part]];
			int index = 0;
			int did = 0;
			int vid = 0;
			int stopIndex = 0;
			for (int i = filePosition; i < fileLines.length; ++i) {
				String line = fileLines[i];
				String[] tokens = line.split("[\t]");
				int tokenCount = tokens.length;
				if (tokenCount != 4) {
					System.out.println("Error: tokenCount not equal to 4 at " + line);
					continue;
				}
				String doc = tokens[0];
				String word = tokens[1];
				String occurs = tokens[2];
				String docTime = tokens[3];
				int time = Integer.parseInt(docTime);
				did = docTypes.indexOf(doc);
				if (did > partitionCuts[part]) {
					filePosition = i;
					break;
				}
				vid = wordTypes.indexOf(word);
				wordCount = Integer.parseInt(occurs);
				stopIndex = index + wordCount;
				wordCounts[vid] += wordCount;
				if (stopIndex >= partitionSizes[part]) {
					for (int j = index; j < partitionSizes[part]; ++j) {
						docIDs[j] = did;
						wordIDs[j] = vid;
						timestamps[j] = time;
					}
					continue;
				}
				fill(docIDs, index, stopIndex + 1, did);
				fill(wordIDs, index, stopIndex + 1, vid);
				fill(timestamps, index, stopIndex + 1, time);
				index = stopIndex;
			}
		int[] randomTopics = new int[partitionSizes[part]];
		for (int i = 0; i < partitionSizes[part]; ++i) {
			randomTopics[i] = randomize.nextInt(numTopics);
		}
		int startDocument = 0;
		if (part > 0) {
			startDocument = partitionCuts[part - 1];
		}
		int[] parameters = {startDocument, partitionCuts[part], V, partitionSizes[part], numTopics, minTime, maxTime};
		partitions[part] = new Gibbs(wordIDs, docIDs, randomTopics, timestamps, parameters);
		}
	}
	
	public void exportTopics(String outPath, int[][] topicArray) {
		int Z = topicArray.length;
		ArrayList<String> export = new ArrayList<String>();
		for (int t = 0; t < Z; ++t) {
			export.add("Topic " + t);
			for (int ID : topicArray[t]) {
				export.add(wordTypes.get(ID));
			}
			export.add("----------");
		}
		LineWriter outFile = new LineWriter(outPath, false);
		String[] exportArray = export.toArray(new String[export.size()]);
		outFile.send(exportArray);
	}
	
	public void exportWords(String outPath) {
		ArrayList<String> export = new ArrayList<String>();
		for (int w = 0; w < V; ++w) {
			export.add(w + ": " + wordTypes.get(w) + ": " + wordCounts[w]);
		}
		LineWriter outFile = new LineWriter(outPath, false);
		String[] exportArray = export.toArray(new String[export.size()]);
		outFile.send(exportArray);
	}

	public void exportDocs(String outPath) {
		ArrayList<String> export = new ArrayList<String>();
		for (int doc = 0; doc < D; ++doc) {
			export.add(docTypes.get(doc));
		}
		LineWriter outFile = new LineWriter(outPath, false);
		String[] exportArray = export.toArray(new String[export.size()]);
		outFile.send(exportArray);
	}

}
