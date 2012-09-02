package LDA;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Gibbs {
	int startDoc;
	int endDoc;
	int[] vid;
	int[] did;
	int[] zid;
	int[] timestamp;
	int n;
	static int v;
	int d;
	static int z;
	int minTime;
	int maxTime;
	int timespan;
	int[][] VinT;
	int[][] TinD;
	int[] Zct;
	int[] docSize;
	int[] wordSize;
	int[] timeTotals;
	double[][] timeDist;
	double perplexity;
	static Random randomize = new Random();

public Gibbs(){
	// empty constructor method so we can create an array of Gibbs-capable partitions
}
	
public Gibbs(int[] words, int[] docs, int[] topics, int[] timestamp, int[] parameters){
	startDoc = parameters[0];
	endDoc = parameters[1];
	d = (endDoc - startDoc) + 1;
	v = parameters[2];
	n = parameters[3];
	z = parameters[4];
	minTime = parameters[5];
	maxTime = parameters[6];
	timespan = (maxTime - minTime) + 1;
	vid = words;
	did = docs;
	zid = topics;
	this.timestamp = timestamp;
	perplexity = 0.0;
	
	VinT = new int[v][z];
	for (int[] row : VinT){
		Arrays.fill(row, 0);
	}
	
	TinD = new int[z][d];
	for (int[] row: TinD){
		Arrays.fill(row, 0);
	}
	Zct = new int[z];
	Arrays.fill(Zct, 0);
	
	timeTotals = new int[timespan];
	Arrays.fill(timeTotals, 0);
	
	docSize = new int[d];
	Arrays.fill(docSize, 0);
	
	wordSize = new int[v];
	Arrays.fill(wordSize, 0);
	
	int[][] topicTimes = new int[z][timespan];
	for (int t = 0; t < z; ++t) {
		Arrays.fill(topicTimes[t], 0);
	}
	
	// randomizes the sequence of tokens
	int temp = 0;
	for (int i = 0; i < n; ++i) {
		int randomPos = randomize.nextInt(n);
		temp = did[i];
		did[i] = did[randomPos];
		did[randomPos] = temp;
		temp = vid[i];
		vid[i] = vid[randomPos];
		vid[randomPos] = temp;
		temp = timestamp[i];
		timestamp[i] = timestamp[randomPos];
		timestamp[randomPos] = temp;
	}

	// constructs arrays and counts based on the raw sequences
	for (int i = 0; i < n; ++i){
		try{
		did[i] = did[i] - startDoc;
		// Makes all docIDs in this partition relative to startDoc.
		++VinT[vid[i]][zid[i]];
		++TinD[zid[i]][did[i]];
		++Zct[zid[i]];
		++docSize[did[i]];
		++wordSize[vid[i]];
		int startPt = timestamp[i] - 5;
		int endPt = timestamp[i] + 5;
		if (startPt < minTime) {
			startPt = minTime;
		}
		if (endPt > maxTime) {
			endPt = maxTime;
		}
		startPt -= minTime;
		endPt -= minTime;
		for (int j = startPt; j < endPt + 1; ++j) {
			++timeTotals[j];
			++topicTimes[zid[i]][j];
		}
		}

		catch (ArrayIndexOutOfBoundsException e){
			System.out.println("i" + "=" + i + " / n = " + n);
			System.out.println("time" + "=" + timestamp[i]);
			System.out.println("zid" + "=" + zid[i] + " / z = " + z);
			System.out.println("vid = " + vid[i] + " / v =" + v);
			System.out.println("did = " + did[i] + " / d =" + d);
		}
	}
	timeDist = new double [z][timespan];
	for (int t = 0; t < z; ++ t) {
		for (int year = 0; year < timespan; ++year) {
			timeDist[t][year] = Math.log((((topicTimes[t][year] + 10) / ( (double) Zct[t] + 10)) * 100) + 5);
		}
	}
}

public void cycle(int[][] typesinTopic, int[] topicTotal){
	for (int i = 0; i < n; ++i){
		int t = zid[i];
		-- TinD[t][did[i]];
		-- VinT[vid[i]][t];
		-- typesinTopic[vid[i]][t];
		-- Zct[t];
		-- topicTotal[t];
		double[] Prob = new double[z];
		for (int top = 0; top < z; ++top){
			Prob[top] = (( (typesinTopic[vid[i]][top] + .05) * (TinD[top][did[i]] + 1) ) / (double) (topicTotal[top] + d));
		}
		Multinomial distribution = new Multinomial(Prob);
		t = distribution.sample();
		zid[i] = t;
		++ TinD[t][did[i]];
		++ VinT[vid[i]][t];
		++ typesinTopic[vid[i]][t];
		++ Zct[t];
		++ topicTotal[t];
	}
	int[][] topicTimes = new int[z][timespan];
	for (int t = 0; t < z; ++t) {
		Arrays.fill(topicTimes[t], 0);
	}
	for (int i = 0; i < n; ++i){
		int startPt = timestamp[i] - 5;
		int endPt = timestamp[i] + 5;
		if (startPt < minTime) {
			startPt = minTime;
		}
		if (endPt > maxTime) {
			endPt = maxTime;
		}
		startPt -= minTime;
		endPt -= minTime;
		for (int j = startPt; j < endPt + 1; ++j) {
			++topicTimes[zid[i]][j];
		}
	}
	for (int t = 0; t < z; ++ t) {
		for (int year = 0; year < timespan; ++year) {
			timeDist[t][year] = Math.log((((topicTimes[t][year] + 10) / ( (double) Zct[t] + 10)) * 100) + 5);
		}
	}
}

public String topicmap(){
	String map = "Topic One";
	for (int i = 0; i < v; ++i){
		int rounded = (int) (VinT[i][0]);
		map = map + " " + rounded;
	}
	map = map + "\nTopic Two:";
	for (int i = 0; i < v; ++i){
		int rounded = (int) (VinT[i][1]);
		map = map + " " + rounded;
	}
	map = map + "\nTopic Three:";
	for (int i = 0; i < v; ++i){
		int rounded = (int) (VinT[i][2]);
		map = map + " " + rounded;
	}
	return map + "\n";
}
	
public void perplex() {
	double sumLog = 0;
	for (int i = 0; i < n; ++i) {
		int word = vid[i];
		int doc = did[i];
		int t = zid[i];
		double prob = ((TinD[t][doc] + 1) / (double) (docSize[doc] + d)) * ((VinT[word][t] + 1) / (double) (Zct[t] + v));
		sumLog += Math.log(prob);
	}
	sumLog = 0-(sumLog / (double) n);
	perplexity = Math.exp(sumLog);
}

public int[][] topics(int topNwords) {
	// Extracts a visualization of each topic according to the formula in
	// Blei and Lafferty, 2009. Essentially, it ranks words by this weighting:
	// probability in topic * log (prob in topic / geometric mean prob in all topics)
	// We begin by converting the integer counts to a probability matrix, phi.
	
	double[][] phi = this.phi();
	
	// Then we calculate the geometric mean of each row in phi, which
	// is a vector of per-topic probabilities for a specific word.
	
	double[] geomMean = new double[v];
	for (int w = 0; w < v; ++w) {  
		// double prod = 1.0;
		// for (double prob : phi[w]) { 
		// 	prod *= prob; 
		// }
		// double inverseZ = 1 / (double) z;
		// geomMean[w] = Math.pow(prod, inverseZ);
		double logSum = 0.0;
		for (double prob: phi[w]) {
			logSum += Math.log(prob);
		}
		geomMean[w] = Math.exp(logSum/z);
	}
	
	// The method is going to return a matrix listing the indices of
	// the top-ranked N words in each topic. To achieve that it produces
	// a vector of weighted scores for words in each topic and then sends
	// them to a method that simultaneously sorts scores and original indices.
	// Note that we are now proceeding through phi column-wise.
	
	int[][] topicArray = new int [z][topNwords];
	for (int t = 0; t < z; ++t) {
		double[] weightedTopic = new double[v];
		for (int w = 0; w < v; ++w) {
			weightedTopic[w] = phi[w][t] * Math.log(phi[w][t] / geomMean[w]);
		}
		int[] ranking = ranks(weightedTopic);
		// for (int i = 0; i < v; ++i) {
		// 	System.out.print(ranking[i] + "," + weightedTopic[i] + " ");
		// }
		// System.out.print("\n");
		topicArray[t] = Arrays.copyOf(ranking, topNwords);
	}
	return topicArray;
}

protected double[][] phi() {
	// turns the matrix of words-in-topics into a probability
	// matrix describing the probability that a given word will be produced
	// by a given topic
	double[][] phiArray = new double [v][z];
	for (int t = 0; t < z; ++t) {
		for (int w = 0; w < v; ++ w) {
			phiArray[w][t] = (VinT[w][t] + 1) / (double) (Zct[t] + 1);     // # of this word in this topic over total # in topic
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

public void exportVinT(String outPath) {
	ArrayList<String> export = new ArrayList<String>();
	for (int t = 0; t < z; ++t) {
		String outLine = t + ",";
		for (int w = 0; w < v; ++w) {
			outLine = outLine + VinT[w][t] + ",";
		}
		export.add(outLine + "\n");
	}
	LineWriter outFile = new LineWriter(outPath, false);
	String[] exportArray = export.toArray(new String[export.size()]);
	outFile.send(exportArray);
}

public void exportTimeline(String outPath) {
	
	ArrayList<String> export = new ArrayList<String>();
	for (int t = 0; t < z; ++t){
		for (int year = 0; year < timespan; ++ year) {
			String outLine = t + ": " + year + ": " + timeDist[t][year];
			export.add(outLine + "\n");
		}
	}
	LineWriter outFile = new LineWriter(outPath, false);
	String[] exportArray = export.toArray(new String[export.size()]);
	outFile.send(exportArray);
}

public void topicTimelines(String outPath) {
	
	ArrayList<String> export = new ArrayList<String>();
	
	int[][] topicTimes = new int[z][timespan];
	for (int t = 0; t < z; ++t) {
		Arrays.fill(topicTimes[t], 0);
	}
	double[][] topicProminence = new double[z][timespan];
	for (int t = 0; t < z; ++t) {
		Arrays.fill(topicProminence[t], 0);
	}
	
	for (int i = 0; i < n; ++i){
		int startPt = timestamp[i] - 5;
		int endPt = timestamp[i] + 5;
		if (startPt < minTime) {
			startPt = minTime;
		}
		if (endPt > maxTime) {
			endPt = maxTime;
		}
		startPt -= minTime;
		endPt -= minTime;
		for (int j = startPt; j < endPt + 1; ++j) {
			++topicTimes[zid[i]][j];
		}
	}
	for (int t = 0; t < z; ++ t) {
		for (int year = 0; year < timespan; ++year) {
			topicProminence[t][year] = (topicTimes[t][year] + 1) / ( (double) timeTotals[t] + 1);
		}
	}
	
	for (int t = 0; t < z; ++t){
		String outLine = Double.toString(topicProminence[t][0]);
		for (int year = 1; year < timespan; ++ year) {
			outLine = outLine + "," + Double.toString(topicProminence[t][year]);
		}
		export.add(outLine);
	}
	
	LineWriter outFile = new LineWriter(outPath, false);
	String[] exportArray = export.toArray(new String[export.size()]);
	outFile.send(exportArray);
}

}
