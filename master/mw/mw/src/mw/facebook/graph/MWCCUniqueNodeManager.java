package mw.facebook.graph;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;


public class MWCCUniqueNodeManager {

	private static Map<String, MWCCUniqueNode> knownUniqueNodes = new HashMap<String, MWCCUniqueNode>();
	
	
	public static Collection<MWCCUniqueNode> listUniqueNodes() {
		return knownUniqueNodes.values();
	}
	
	public static boolean existsUniqueNode(String name) {
		return (knownUniqueNodes.get(name) != null);
	}
	
	public static synchronized MWCCUniqueNode getUniqueNode(String name) {
		MWCCUniqueNode uniqueNode = knownUniqueNodes.get(name);
		if(uniqueNode == null) {
			uniqueNode = new MWCCUniqueNode(name);
			knownUniqueNodes.put(name, uniqueNode);
		}
		return uniqueNode;
	}
	
	
	public static void storeSingleUniqueNode(MWCCUniqueNode node, DataOutputStream stream) throws IOException {
		String idName = node.getID();
		stream.writeInt(idName.length());
		stream.writeBytes(idName);
	}
	
	public static MWCCUniqueNode loadSingleUniqueNode(DataInputStream stream) throws IOException {
		int idNameLength = stream.readInt();
		byte[] idNameBytes = new byte[idNameLength];
		stream.read(idNameBytes);
		return getUniqueNode(new String(idNameBytes));
	}

	
	public static void storeUniqueNodes(Collection<MWCCUniqueNode> nodes, DataOutputStream stream) throws IOException {
		stream.writeInt(nodes.size());
		for(MWCCUniqueNode node: nodes) {
			storeSingleUniqueNode(node, stream);
		}
	}
	
	public static void loadUniqueNodes(Collection<MWCCUniqueNode> nodes, DataInputStream stream) throws IOException {
		int n = stream.readInt();
		for(int i = 0; i < n; i++) {
			MWCCUniqueNode uniqueNode = loadSingleUniqueNode(stream);
			nodes.add(uniqueNode);
		}
	}

}
