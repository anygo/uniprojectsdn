package mw.facebook.graph;



public class MWCCRealName extends MWCCDataNode {

	private static final String REAL_NAME_TAG = "REAL_NAME_TAG";
	
	
	private final String realName;
	

	public MWCCRealName(String realName) {
		super(null, REAL_NAME_TAG);
		this.realName = realName;
	}
	
	
	public static MWCCRealName getData(MWCCUniqueNode uniqueNode) {
		return (MWCCRealName) uniqueNode.getDataNode(REAL_NAME_TAG);
	}
	
	
	public String getRealName() {
		return realName;
	}
	

	@Override
	public String toString() {
		return realName;
	}
	
}
