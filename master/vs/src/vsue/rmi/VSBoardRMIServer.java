package vsue.rmi;
import java.rmi.AlreadyBoundException;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;


public class VSBoardRMIServer {

	public static void main(String args[]) throws RemoteException, AlreadyBoundException
	{
		// Remote-Objekt exportieren
		VSBoard b = new VSBoardImpl();
		VSBoard board = (VSBoard) UnicastRemoteObject.exportObject(b, 0);
		
		// Remote-Objekt bekannt machen
		Registry registry = LocateRegistry.createRegistry(12345);
		registry.bind("board", board);
	}
	
}
