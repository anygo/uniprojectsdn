package vsue.proxy;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;

public class VSCommunication {
	
	public VSConnection openConnection(String host, int port) {

		Socket socket = new Socket();
		try {
			
			socket.connect(new InetSocketAddress(host, port));
			return new VSConnection(socket);
			
		} catch (IOException e) {
			return null;
		}
		
	}

	public void closeConnection(VSConnection connection) {

		Socket socket = connection.getSocket();
		try {
			
			socket.close();
			
		} catch (IOException e) {
			return;
		}

	}

	public void createServer(VSServer server, int port) {

		/**
		 * Create VSServerThread
		 *   VSServerThread:
		 *     while(true)
		 *       Socket socket = serversocket.accept()
		 *       VSConnection connection = new VSConnection(socket)
		 *       Create VSHandleConnectionThread
		 *         VSHandleConnectionThread:
		 *           call server.handleRequest(connection)
		 */
		
		try {
			
			ServerSocket serversocket = new ServerSocket(port);
			VSServerThread serverthread = new VSServerThread(server, serversocket);
			Thread thread = new Thread(serverthread);
			thread.start();
			
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}

