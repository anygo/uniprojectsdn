package vsue.communication;

import java.io.IOException;
import java.net.BindException;
import java.net.ServerSocket;
import java.util.HashMap;

import vsue.VSConstants;
import vsue.replica.Box;


public abstract class VSServer implements Runnable {
	private int port;

	public VSServer() {
		port = VSConstants.VSSERVER_STARTING_PORT;
	}

	public int getPort() {
		return port;
	}

	private ServerSocket createServerSocket() throws IOException {
		ServerSocket socket = null;
		boolean exception = false;
		int counter = 0;

		do {
			counter++;
			exception = false;
			try {
				socket = new ServerSocket(getPort());
			} catch (BindException e) {
				port++;
				exception = true;
				System.out.println("Default port already used, trying " + port);
			}
		} while (exception && counter < 20);

		return socket;
	}

	public void run() {
		ServerSocket socket = null;
		HashMap<String, Box> map = null;
		
		try {
			socket = createServerSocket();
			if (socket == null) {
				System.out.println("No unused port");
				return;
			}
			map = new HashMap<String, Box>();

			System.out.println("VSServer is running...");
			while (true) {
				VSObjectConnection vsoCon = new VSObjectConnection(
						new VSConnection(socket.accept()));
				
				startExecution(vsoCon, map);
				
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public abstract void startExecution(VSObjectConnection vsoCon, HashMap<String, Box> map) throws Exception;
}
