package vsue.communication;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;

import vsue.VSConstants;

public class VSConnection {
	private Socket m_sock;
	private OutputStream m_os;
	private InputStream m_is;

	public VSConnection(Socket socket) throws IOException {
		socket.setTcpNoDelay(true);
		
		m_sock = socket;
		m_os = m_sock.getOutputStream();
		m_is = m_sock.getInputStream();
	}

	public void close() throws Exception {
		m_os.close();
		m_is.close();
		m_sock.close();
	}
	
	public void sendChunk(byte[] chunk) throws IOException {		
		int size = chunk.length;
		
		m_os.write(size);
		m_os.write(size >> 8);
		m_os.write(size >> 16);
		m_os.write(size >> 24);
		m_os.write(chunk);
	}

	public synchronized byte[] receiveChunk() throws IOException {
		int size = 0;
		int size1 = 0;
		int size2 = 0;
		int size3 = 0;
		int size4 = 0;

		if (m_sock.isClosed())
		{	
			throw new IOException(VSConstants.CLOSED_MESSAGE);
		}

		size1 = m_is.read();
		size2 = m_is.read();
		size3 = m_is.read();
		size4 = m_is.read();

		if (size1 == -1 || size2 == -1 || size3 == -1 || size4 == -1) {
			throw new IOException(VSConstants.CLOSED_MESSAGE);
		}

		size = size4 << 8;
		size = size | size3;
		size = size << 8;
		size = size | size2;
		size = size << 8;
		size = size | size1;

		byte[] ret = new byte[size];
		
		for (int i = 0; i < size; i++) {
			ret[i] = (byte) m_is.read();
		}

		return ret;
	}
}
