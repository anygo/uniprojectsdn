package mw.facebook.graph;

import java.util.HashMap;
import java.util.Map;


public class MWCCUniqueNode {

	private final String id;
	private final transient Map<String, MWCCDataNode> dataNodes;
	
	
	MWCCUniqueNode(String id) {
		this.id = id;
		dataNodes = new HashMap<String, MWCCDataNode>();
	}

	
	public String getID() {
		return id;
	}

	
	public void addDataNode(String key, MWCCDataNode dataNode) {
		dataNodes.put(key, dataNode);
	}
	
	public MWCCDataNode getDataNode(String key) {
		return dataNodes.get(key);
	}
	

	@Override
	public boolean equals(Object obj) {
		if(!(obj instanceof MWCCUniqueNode)) return false;
		return (this == obj);
	}
	
	@Override
	public String toString() {
		return id;
	}

}
