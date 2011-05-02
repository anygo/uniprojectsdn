package vs;

import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.Vector;

public interface IBoard extends Remote {

	public void post(Message msg) throws RemoteException;
	public Vector<Message> get(int count) throws RemoteException;
	public void listen(ICallbackHandler cbh) throws RemoteException;
}

