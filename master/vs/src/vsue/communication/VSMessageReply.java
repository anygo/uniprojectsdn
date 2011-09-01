package vsue.communication;

import java.io.Serializable;

public class VSMessageReply implements Serializable {
	private static final long serialVersionUID = 1L;
	
	private Object returnValue;
	private long remoteCallID;
	private int sequenceNumber;
	
	public VSMessageReply() {
		this(0, 0, null);
	}
	
	public VSMessageReply(long remoteCallID, int sequenceNumber, Object returnValue) {
		this.remoteCallID = remoteCallID;
		this.sequenceNumber = sequenceNumber;
		this.returnValue = returnValue;
	}

	public Object getReturnValue() {
		return returnValue;
	}

	public long getRemoteCallID() {
		return remoteCallID;
	}

	public int getSequenceNumber() {
		return sequenceNumber;
	}
	
	public String toString() {
		return "RemoteCallID: " + remoteCallID + ", sequenceNumber: " + sequenceNumber;
	}
}
