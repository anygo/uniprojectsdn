package vs;

import java.rmi.RemoteException;
import java.rmi.registry.Registry;
import java.rmi.registry.LocateRegistry;
import java.rmi.server.UnicastRemoteObject;
import java.lang.SecurityManager;
import java.util.Vector;
import java.util.Collections;
import java.util.Iterator;

public class boardserver implements IBoard {

	private Vector<Message> m_msgVector;
	private Vector<ICallbackHandler> m_cbhVector;

	public boardserver() {

		m_msgVector = new Vector<Message>();
		m_cbhVector = new Vector<ICallbackHandler>();
	}

	public void post(Message msg) throws RemoteException {

		m_msgVector.add(msg);

		Iterator<ICallbackHandler> it = m_cbhVector.iterator();
		while (it.hasNext()) {

			try {
				it.next().callback(msg);
			}
			catch (RemoteException e) {
				// exception might be thrown
				// if client disconnects
				it.remove();
			}
		}
	}

	public Vector<Message> get(int count) throws RemoteException {

		if (count <= 0)
			return null;

		Vector<Message> resVector;
		int vecSize = m_msgVector.size();
		int toCopy = vecSize;
		
		if (count < toCopy)
			toCopy = count;

		// copy last toCopy elements in new vector
		resVector = new Vector<Message>(
			m_msgVector.subList(vecSize - toCopy, vecSize)
			);

		// reverse order so the newest will be displayed first
		Collections.reverse(resVector);

		return resVector;
	}

	public void listen(ICallbackHandler cbh) throws RemoteException {

		m_cbhVector.add(cbh);
	}

	public static void main(String argv[]) {

		if (System.getSecurityManager() == null) {
			System.setSecurityManager(new SecurityManager());
		}

		try {

			Registry registry;
			IBoard stub;
			IBoard boardObj = new boardserver();
			String objName = "board";

			stub = (IBoard)
				UnicastRemoteObject.exportObject(boardObj, 0);

			registry = LocateRegistry.getRegistry(10412);
			registry.rebind(objName, stub);

		} catch (Exception e) {

			e.printStackTrace();
		}
	}
}

