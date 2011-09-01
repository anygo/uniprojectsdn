package vsue.rmi;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.rmi.RemoteException;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import vsue.replica.VSRemoteObjectStateHandler;

public class VSBoardImpl implements VSBoard, VSRemoteObjectStateHandler {

	private LinkedList<VSBoardMessage> board;
	private List<VSBoardListener> listeners;

	public VSBoardImpl() {
		board = new LinkedList<VSBoardMessage>();
		listeners = Collections
				.synchronizedList(new LinkedList<VSBoardListener>());
	}

	@Override
	public void post(VSBoardMessage message) throws RemoteException {
		board.addFirst(message);
		informListeners(message);
	}

	@Override
	public VSBoardMessage[] get(int n) throws IllegalArgumentException,
			RemoteException {
		if (n < 0) {
			throw new IllegalArgumentException();
		}

		if (n >= board.size())
			n = board.size();

		VSBoardMessage[] arr = new VSBoardMessage[n];
		for (int i = 0; i < n; ++i) {
			arr[i] = board.get(board.size() - i - 1);
		}
		
		return arr;
	}

	@Override
	public void listen(VSBoardListener listener) throws RemoteException {
		listeners.add(listener);
	}

	private void informListeners(VSBoardMessage message) {
		Iterator<VSBoardListener> it = listeners.iterator();
		
		while (it.hasNext()) {
			try {
				System.out.println("Informing clients");
				it.next().newMessage(message);
			} catch (RemoteException e) {
				System.out
						.println("Listener error - removing this listener from list...");
				it.remove();
			}
		}
	}

	@Override
	public byte[] getState() {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ObjectOutputStream oos;
		try {
			oos = new ObjectOutputStream(baos);
			oos.writeObject(board);
			oos.writeObject(listeners);
			oos.close();
		} catch (IOException e1) {
			e1.printStackTrace();
			return null;
		}	
		return baos.toByteArray();
	}

	@SuppressWarnings("unchecked")
	@Override
	public void setState(byte[] state) {
		ByteArrayInputStream bais = new ByteArrayInputStream(state);
		ObjectInputStream ois;
		
		try {
			ois = new ObjectInputStream(bais);
			board = (LinkedList<VSBoardMessage>) ois.readObject();
			listeners = (List<VSBoardListener>) ois.readObject();	
			ois.close();
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
}
