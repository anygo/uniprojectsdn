package mw.facebook.graph;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;



public class MWCCNeighborhoodUtils {

	/*
	 * Graph construction
	 */
	
	public static boolean createNeighborLink(MWCCUniqueNode nodeA, MWCCUniqueNode nodeB) {
		MWCCNeighborhood neighborhoodA = MWCCNeighborhood.getData(nodeA);
		if(!neighborhoodA.addNeighbor(nodeB)) return false;

		MWCCNeighborhood neighborhoodB = MWCCNeighborhood.getData(nodeB);
		return neighborhoodB.addNeighbor(nodeA);
	}


	/*
	 * Graph analysis
	 */
	
	public static Collection<MWCCUniqueNode> getNeighbors(MWCCUniqueNode node, int depth) {
		if(depth == 0) {
			Set<MWCCUniqueNode> me = new HashSet<MWCCUniqueNode>();
			me.add(node);
			return me;
		} else {
			Set<MWCCUniqueNode> allNeighbors = new HashSet<MWCCUniqueNode>();
			allNeighbors.addAll(MWCCNeighborhood.getData(node).getNeighbors());
			
			MWCCNeighborhood neighborhood = MWCCNeighborhood.getData(node);
			for(MWCCUniqueNode neighbor: neighborhood.getNeighbors()) {
				Collection<MWCCUniqueNode> neighborNeighbors = getNeighbors(neighbor, depth - 1);
				allNeighbors.addAll(neighborNeighbors);
			}
			
			return allNeighbors;
		}
	}

	public static List<MWCCUniqueNode> getShortestPath(Collection<MWCCUniqueNode> graph, MWCCUniqueNode nodeA, MWCCUniqueNode nodeB) {
		// Dijkstra's algorithm

		// Initialization
		List<MWCCDijkstraNode> remainingNodes = new LinkedList<MWCCDijkstraNode>();
		for(MWCCUniqueNode node: graph) {
			if(node == nodeA) continue;
			remainingNodes.add(new MWCCDijkstraNode(node));
		}
		
		MWCCDijkstraNode dNodeA = new MWCCDijkstraNode(nodeA);
		dNodeA.setDistance(0);
		remainingNodes.add(0, dNodeA);
		

		// Create predecessor information
		MWCCDijkstraNode dNodeB = null;
		
		while(!remainingNodes.isEmpty()) {
			Collections.sort(remainingNodes);
			
			MWCCDijkstraNode dNode = remainingNodes.remove(0);
			MWCCUniqueNode node = dNode.getUniqueNode();

			if(node.equals(nodeB)) {
				dNodeB = dNode;
			}
			
			MWCCNeighborhood neighborhood = MWCCNeighborhood.getData(node);
			for(MWCCUniqueNode neighbor: neighborhood.getNeighbors()) {
				MWCCDijkstraNode dNeighbor = null;
				for(MWCCDijkstraNode dn: remainingNodes) {
					if(dn.getUniqueNode().equals(neighbor)) {
						dNeighbor = dn;
						break;
					}
				}
				if(dNeighbor == null) continue;
				
				int alternative = dNode.getDistance() + 1;
				if(alternative < dNeighbor.getDistance()) {
					dNeighbor.setDistance(alternative);
					dNeighbor.setPredecessor(dNode);
				}
			}
		}

		
		// Extract shortest path
		List<MWCCUniqueNode> shortestPath = new LinkedList<MWCCUniqueNode>();
		shortestPath.add(nodeB);
		
		MWCCDijkstraNode dn = dNodeB;
		while(dn.getPredecessor() != null) {
			dn = dn.getPredecessor();
			shortestPath.add(0, dn.getUniqueNode());
		}
		
		return shortestPath;
	}

	public static List<MWCCUniqueNode> getPath(MWCCUniqueNode nodeA, MWCCUniqueNode nodeB) {
		Collection<MWCCUniqueNode> neighborhoodA;
		Collection<MWCCUniqueNode> neighborhoodB;

		int depth = 0;
		do {
			depth++;
			System.out.println("SEARCHING: depth "+ depth);
			neighborhoodA = getNeighbors(nodeA, depth);
			neighborhoodB = getNeighbors(nodeB, depth);
		} while(Collections.disjoint(neighborhoodA, neighborhoodB));
//		System.out.println("FOUND: neighborhoods overlap in depth " + depth);
		
		Set<MWCCUniqueNode> combinedNeighborhoods = new HashSet<MWCCUniqueNode>();
		combinedNeighborhoods.addAll(neighborhoodA);
		combinedNeighborhoods.addAll(neighborhoodB);
		
		return getShortestPath(combinedNeighborhoods, nodeA, nodeB);
	}
	
	
	/*
	 * Graph visualization
	 */
	
	public static void printNeighborhood(MWCCUniqueNode node, int depth) {
		System.out.println("digraph G {\n");
		
		MWCCNeighborhood neighborhood = MWCCNeighborhood.getData(node);
		neighborhood.printNeighbors();
		
		Collection<MWCCUniqueNode> neighbors = getNeighbors(node, depth);
		for(MWCCUniqueNode neighbor: neighbors) {
			MWCCNeighborhood neighborNeighborhood = MWCCNeighborhood.getData(neighbor);
			neighborNeighborhood.printNeighbors();
		}
		
		System.out.println("}");
	}

	
	/*
	 * Graph dump
	 */

	public static void storeNeighborhoods(Collection<MWCCUniqueNode> nodes, DataOutputStream stream) throws IOException {
		stream.writeInt(nodes.size());
		for(MWCCUniqueNode node: nodes) {
			MWCCNeighborhood neighborhood = MWCCNeighborhood.getData(node);
			neighborhood.store(stream);
		}
	}

	public static void loadNeighborhoods(DataInputStream stream) throws IOException {
		int n = stream.readInt();
		for(int i = 0; i < n; i++) {
			MWCCNeighborhood neighborhood = new MWCCNeighborhood();
			neighborhood.load(stream);
			neighborhood.addData(neighborhood.getUniqueNode());
		}
	}

}
