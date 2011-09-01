package vsue.rpc;

import java.rmi.Remote;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

import vsue.VSConstants;
import vsue.rmi.VSBoard;
import vsue.rmi.VSBoardImpl;

public class VSBoardServer {	
	public static void main(String args[]) {
		VSBoard boardObject = null;
		Remote boardRemote = null;
		Registry registry = null;

		try {
			boardObject = new VSBoardImpl();
			boardRemote = VSRemoteObjectManager.getInstance().exportObject(
					boardObject);

			registry = LocateRegistry.createRegistry(VSConstants.REGISTRY_PORT_BOARD_SERVER);
			registry.bind("board", boardRemote);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
