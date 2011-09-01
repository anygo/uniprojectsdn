package vsue.rmi;
import java.rmi.Remote;
import java.rmi.RemoteException;


public interface VSBoard extends Remote {
	public void post(VSBoardMessage message) throws RemoteException;
	public VSBoardMessage[] get(int n) throws IllegalArgumentException, RemoteException;
	public void listen(VSBoardListener listener) throws RemoteException;
}
