package vsue.callback;

import vsue.proxy.VSClient;
import vsue.proxy.VSServer;

public class VSRemoteEntity {
	
	VSClient m_client;
	VSServer m_server;

	public VSRemoteEntity() {
		
		m_client = new VSClient();
		m_server = new VSServer();
		
	}
	
	public void init(int serverPort) {
		
		m_client.init();
		m_server.init(serverPort);
		
	}
	
	public void exportObject(Object object) {
		
		m_server.exportObject(object);
		
	}
	
	public Object lookup(String host, int port, Class interfaceClass) {
		
		return m_client.lookup(host, port, interfaceClass);
		
	}
	
}


