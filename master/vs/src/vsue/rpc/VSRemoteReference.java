package vsue.rpc;

import java.io.Serializable;

@SuppressWarnings("serial")
public class VSRemoteReference implements Serializable {
	private String host;
	private int port;
	private int objectID;
	
	public VSRemoteReference() {
		host = null;
		port = 0;
		objectID = 0;
	}
	
	public VSRemoteReference(String host, int port, int objectID) {
		this.host = host;
		this.port = port;
		this.objectID = objectID;
	}

	public String getHost() {
		return host;
	}

	public void setHost(String host) {
		this.host = host;
	}

	public int getPort() {
		return port;
	}

	public void setPort(int port) {
		this.port = port;
	}

	public int getObjectID() {
		return objectID;
	}

	public void setObjectID(int objectID) {
		this.objectID = objectID;
	}
}

