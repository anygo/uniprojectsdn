package mw.facebook.graph;



public class MWCCFriendCount extends MWCCDataNode {

	private static final String FRIEND_COUNT_TAG = "FRIEND_COUNT_TAG";
	
	
	private final int friendCount;
	

	public MWCCFriendCount(int friendCount) {
		super(null, FRIEND_COUNT_TAG);
		this.friendCount = friendCount;
	}
	
	
	public static MWCCFriendCount getData(MWCCUniqueNode uniqueNode) {
		return (MWCCFriendCount) uniqueNode.getDataNode(FRIEND_COUNT_TAG);
	}


	public int getFriendCount() {
		return friendCount;
	}
	

	@Override
	public String toString() {
		return String.valueOf(friendCount);
	}
	
}
