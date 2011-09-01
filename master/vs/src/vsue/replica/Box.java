package vsue.replica;

public class Box {
	public Object object;
	public long remoteCallID;

	public Box(Object object, long remoteCallID) {
		this.object = object;
		this.remoteCallID = remoteCallID;
	}
}
