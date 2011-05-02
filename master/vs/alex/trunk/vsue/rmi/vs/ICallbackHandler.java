package vs;

import java.rmi.Remote;
import java.rmi.RemoteException;

public interface ICallbackHandler extends Remote {

	public void callback(Message msg) throws RemoteException;
}

