package vsue.proxy;

import java.io.Serializable;

public class VSRemoteReference implements Serializable {

	private String host;
	private int port;
	private int objectID;

	public VSRemoteReference(String host, int port, int id) {
		
		this.host = host;
		this.port = port;
		this.objectID = id;
		
	}

	public String getHost() {
		
		return host;
		
	}
	
	public int getPort() {
		
		return port;
		
	}

	public int getObjectID() {

		return objectID;

	}

}

