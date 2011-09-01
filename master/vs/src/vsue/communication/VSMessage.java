package vsue.communication;

import java.io.Serializable;

public class VSMessage implements Serializable {
	private static final long serialVersionUID = 1L;
	
	private int objectID;
	private String methodName;
	private Object[] arguments;
	private long remoteCallID;
	private int sequenceNumber;
	private String client;

	public VSMessage() {
		this(0, "", null, "");
	}

	public VSMessage(int objectID, String methodName, Object[] arguments, String client) {
		this.objectID = objectID;
		this.methodName = methodName;
		this.arguments = arguments;
		this.client = client;
		this.remoteCallID = System.currentTimeMillis();		
		this.sequenceNumber = 0;
	}
	
	public VSMessage(VSMessage message) {
		this(message.getObjectID(), message.getMethodName(), message.getArguments(), message.getClient());
		this.remoteCallID = message.getRemoteCallID();
		this.sequenceNumber = message.getSequenceNumber();
	}

	public int getObjectID() {
		return objectID;
	}

	public void setObjectID(int objectID) {
		this.objectID = objectID;
	}

	public String getMethodName() {
		return methodName;
	}

	public void setMethodName(String methodName) {
		this.methodName = methodName;
	}

	public Object[] getArguments() {
		return arguments;
	}

	public void setArguments(Object[] arguments) {
		this.arguments = arguments;
	}
	
	public long getRemoteCallID() {
		return remoteCallID;
	}
	
	public int getSequenceNumber() {
		return sequenceNumber;
	}
	
	public void incrementSequenceNumber() {
		sequenceNumber++;
	}
	
	public String toString() {
		return "RemoteCallID: " + remoteCallID + ", sequenceNumber: " + sequenceNumber + ", objectID: " + objectID;
	}

	public void setClient(String client) {
		this.client = client;
	}

	public String getClient() {
		return client;
	}
}
