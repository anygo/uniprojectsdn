package vsue.proxy;

import java.net.InetSocketAddress;

public class VSServer {

	private VSCommunication m_vscom;
	private VSRemoteObjectManager m_manager;

	public VSServer() {

		m_vscom = new VSCommunication();

	}

	public void init(int port) {

		m_vscom.createServer(this, port);
		// TODO use external address instead of localhost
		m_manager = new VSRemoteObjectManager(new InetSocketAddress("localhost", port));

	}

	public void exportObject(Object object) {

		m_manager.exportObject(object);

	}

	public void handleRequest(VSConnection connection) {

		VSMessage result;
		VSMessage msg = connection.receiveMessage();

		// TODO: use isXXObject()
		if(msg.getVSLookupObject() != null) {
			
			VSLookupObject lookup = msg.getVSLookupObject();

			try {
				
				VSRemoteReference remoteref = m_manager.getRemoteReference(lookup.getInterfaceClass());
				result = new VSMessage(remoteref);

			} catch (ClassNotFoundException e) {
				result = new VSMessage(e);
			}

		} else if(msg.getVSMethodObject() != null) {
			
			VSMethodObject mobj = msg.getVSMethodObject();
			try {
				// wrap result in VSResultObject
				VSResultObject tmp = new VSResultObject(m_manager.invokeMethod(mobj.getObjectID(), mobj.getGenericMethodName(), mobj.getArgs()));
				result = new VSMessage(tmp);
			} catch (Exception e) {
				result = new VSMessage(e);
			}

		} else {
			result = new VSMessage(new Exception("handleRequest: unknown request"));
		}

		if(!connection.sendMessage(result)) {
			/*
			 * Failed to send message
			 * 
			 * Possible reasons are:
			 * - message not serializable
			 * - connection lost 
			 */
			System.out.println("DEBUG: handleRequest: failed to send message to client");
		}

	}

}
