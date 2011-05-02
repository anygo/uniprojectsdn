package vsue.proxy;

import java.io.IOException;
import java.io.NotSerializableException;
import java.net.Socket;

import vsue.marshalling.VSObjectInputStream;
import vsue.marshalling.VSObjectOutputStream;

public class VSConnection {
	
	private Socket m_socket;
	private VSObjectInputStream m_is;
	private VSObjectOutputStream m_os;
	
	public VSConnection(Socket socket) throws IOException {
		
		m_socket = socket;
		m_is = new VSObjectInputStream(m_socket.getInputStream());
		m_os = new VSObjectOutputStream(m_socket.getOutputStream());
		
	}
	
	public Socket getSocket() {

		return m_socket;
		
	}

	public boolean sendMessage(VSMessage msg) {

		try {
			m_os.writeObject(msg);
		} catch (Exception e) {
			return false;
		}
		
		return true;

	}

	public VSMessage receiveMessage() {

		try {
			return (VSMessage)m_is.readObject();
		} catch (Exception e) {
			return null;
		}

	}

}

