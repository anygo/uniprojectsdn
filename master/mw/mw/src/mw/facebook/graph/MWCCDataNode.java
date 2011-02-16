package mw.facebook.graph;


public abstract class MWCCDataNode {

	protected final String DATA_NODE_TAG;
	private MWCCUniqueNode uniqueNode;

	
	protected MWCCDataNode(MWCCUniqueNode uniqueNode, String tag) {
		DATA_NODE_TAG = tag;
		this.uniqueNode = uniqueNode;
	}
	
	
	public void addData(MWCCUniqueNode uniqueNode) {
		uniqueNode.addDataNode(DATA_NODE_TAG, this);
		setUniqueNode(uniqueNode);
	}
	
	
	public MWCCUniqueNode getUniqueNode() {
		return uniqueNode;
	}

	public void setUniqueNode(MWCCUniqueNode uniqueNode) {
		this.uniqueNode = uniqueNode;
	}
	
}
