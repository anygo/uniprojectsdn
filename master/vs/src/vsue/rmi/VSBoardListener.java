package vsue.rmi;
import java.rmi.Remote;
import java.rmi.RemoteException;


public interface VSBoardListener extends Remote {
	public void newMessage(VSBoardMessage message) throws RemoteException;
}
