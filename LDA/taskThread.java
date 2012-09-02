package LDA;

class taskThread extends Thread {

	private Gibbs partition;
	private int[][] typesinTopic;
	private int[] topicTotal;
	
	public taskThread(Gibbs partition, int[][] typesinTopic, int[] topicTotal) {
		this.partition = partition;
		this.typesinTopic = typesinTopic;
		this.topicTotal = topicTotal;
		setDaemon(true);
		}

	public void run() {
		// Three tasks in each iteration. First we run Gibbs sampling using current data.
		partition.cycle(typesinTopic, topicTotal);
		
		// then we calculate the perplexity for this partition,
		partition.perplex();
		
	}


}
