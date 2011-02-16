package mw.facebook.graph;



/*
 * Helper class for calculating the shortest path between two nodes.
 */
public class MWCCDijkstraNode implements Comparable<MWCCDijkstraNode> {

	private final MWCCUniqueNode uniqueNode;
	private int distance;
	private MWCCDijkstraNode predecessor;

	
	public MWCCDijkstraNode(MWCCUniqueNode uniqueNode) {
		this.uniqueNode = uniqueNode;
		distance = Integer.MAX_VALUE;
	}


	public int getDistance() {
		return distance;
	}

	public void setDistance(int distance) {
		this.distance = distance;
	}

	public MWCCUniqueNode getUniqueNode() {
		return uniqueNode;
	}

	public MWCCDijkstraNode getPredecessor() {
		return predecessor;
	}

	public void setPredecessor(MWCCDijkstraNode predecessor) {
		this.predecessor = predecessor;
	}


	public int compareTo(MWCCDijkstraNode o) {
		return (distance - o.distance);
	}

	
	@Override
	public boolean equals(Object obj) {
		if(obj instanceof MWCCDijkstraNode) {
			return uniqueNode.equals(((MWCCDijkstraNode) obj).uniqueNode);
		}
		return false;
	}
	
	@Override
	public String toString() {
		return "{" + uniqueNode + "#" + distance + "}";
	}

}
