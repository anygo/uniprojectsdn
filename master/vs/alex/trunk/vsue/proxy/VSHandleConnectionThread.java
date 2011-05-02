package vsue.proxy;

public class VSHandleConnectionThread implements Runnable {

	private VSServer m_server;
	private VSConnection m_connection;
	
	public VSHandleConnectionThread(VSServer server, VSConnection connection) {
		
		m_server = server;
		m_connection = connection;
		
	}
	
	public void run() {
		
		m_server.handleRequest(m_connection);

	}
	
}
