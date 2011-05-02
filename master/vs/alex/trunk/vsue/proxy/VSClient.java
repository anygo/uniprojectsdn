package vsue.proxy;

public class VSClient {

	private VSCommunication m_vcom;
	
	public VSClient() {

		m_vcom = new VSCommunication(); 

	}

	public void init() {

		// This function does nothing!

	}

	public Object lookup(String host, int port, Class interfaceClass) {

		VSRemoteReference remoteref;
		
		// Get VSRemoteReference from server
		VSLookupObject lookupObject = new VSLookupObject(interfaceClass);
		VSMessage msg = new VSMessage(lookupObject); 
		VSConnection lookup = m_vcom.openConnection(host, port);
		
		if(!lookup.sendMessage(msg)) {
			return null;
		}

		msg = lookup.receiveMessage();
		if(msg == null) {
			return null;
		}
		
		// Close connection to server
		m_vcom.closeConnection(lookup);

		// Parse answer from server
		if(msg.getException() != null) {
			return null;
		}
		
		remoteref = msg.getVSRemoteReference();
		if(remoteref == null) {
			return null;
		}

		// Create proxy/invocation handler for remote object
		return VSInvocationHandler.createProxy(remoteref, interfaceClass);
		
	}

}

