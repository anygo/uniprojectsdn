package mw.facebook.graph;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;



public class MWCCNeighborhood extends MWCCDataNode {

	private static final String NEIGHBORHOOD_TAG = "NEIGHBORHOOD_TAG";
	

	private final Set<MWCCUniqueNode> neighbors;
	
	
	public MWCCNeighborhood() {
		this(null);
	}
	
	public MWCCNeighborhood(MWCCUniqueNode uniqueNode) {
		super(uniqueNode, NEIGHBORHOOD_TAG);
		neighbors = new HashSet<MWCCUniqueNode>();
	}
	
	
	public static MWCCNeighborhood getData(MWCCUniqueNode uniqueNode) {
		return (MWCCNeighborhood) uniqueNode.getDataNode(NEIGHBORHOOD_TAG);
	}

	
	public boolean addNeighbor(MWCCUniqueNode neighbor) {
		return neighbors.add(neighbor);
	}

	public Collection<MWCCUniqueNode> getNeighbors() {
		return neighbors;
	}


	@Override
	public String toString() {
		return "{" + getUniqueNode() + ": " + neighbors + "}";
	}
	
	
	public void printNeighbors() {
		System.out.println("\t" + getUniqueNode() + ";");
		for(MWCCUniqueNode neighbor: neighbors) {
			System.out.println("\t" + getUniqueNode() + " -> " + neighbor + " [arrowhead=none,arrowtail=none];");
		}
		System.out.println();
	}

	
	public void store(DataOutputStream stream) throws IOException {
		// Store my unique node info
		MWCCUniqueNodeManager.storeSingleUniqueNode(getUniqueNode(), stream);
		
		// Store the unique node info of my neighbors
		stream.writeInt(neighbors.size());
		for(MWCCUniqueNode neighbor: neighbors) {
			MWCCUniqueNodeManager.storeSingleUniqueNode(neighbor, stream);
		}
	}

	public void load(DataInputStream stream) throws IOException {
		// Load my unique-node info
		setUniqueNode(MWCCUniqueNodeManager.loadSingleUniqueNode(stream));
		
		// Load the unique node info of my neighbors
		int n = stream.readInt();
		for(int i = 0; i < n; i++) {
			MWCCUniqueNode neighbor = MWCCUniqueNodeManager.loadSingleUniqueNode(stream);
			neighbors.add(neighbor);
		}
	}

}
