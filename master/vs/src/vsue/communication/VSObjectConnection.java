package vsue.communication;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class VSObjectConnection {

	private VSConnection m_connection;

	public VSObjectConnection(VSConnection con) {
		m_connection = con;
	}

	public void close() throws Exception {
		m_connection.close();
	}

	public void sendObject(Serializable object) throws IOException {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ObjectOutputStream oos = new ObjectOutputStream(baos);
		oos.writeObject(object);
		
		byte[] bytes = baos.toByteArray();
		m_connection.sendChunk(bytes);
	}

	public Serializable receiveObject() throws IOException, ClassNotFoundException {
		byte[] ret = m_connection.receiveChunk();
		
		if (ret == null) return null;

		ByteArrayInputStream bais = new ByteArrayInputStream(ret);
		ObjectInputStream ois = new ObjectInputStream(bais);

		return (Serializable) ois.readObject();
	}
}
