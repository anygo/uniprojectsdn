package vsue.proxy;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

public class VSServerThread implements Runnable {

	private VSServer m_server;
	private ServerSocket m_serversocket = null;
	
	public VSServerThread(VSServer server, ServerSocket socket) {
	
		m_server = server;
		m_serversocket = socket;
		
	}
	
	public void run() {

		if(m_serversocket == null) {
			return;
		}
		
		while(true) {
			
			try {
				
				Socket socket = m_serversocket.accept();
				VSConnection connection = new VSConnection(socket);
				VSHandleConnectionThread connectionthread = new VSHandleConnectionThread(m_server, connection);
				Thread thread = new Thread(connectionthread);
				thread.start();
				
			} catch (IOException e) {
				
				e.printStackTrace();
				
			}
			
		}

	}

}
