package vs;

import java.rmi.RemoteException;
import java.rmi.registry.Registry;
import java.rmi.registry.LocateRegistry;
import java.rmi.server.UnicastRemoteObject;
import java.rmi.NotBoundException;
import java.rmi.AlreadyBoundException;
import java.rmi.AccessException;
import java.lang.SecurityManager;
import java.util.Vector;
import java.util.Iterator;

public class boardclient implements ICallbackHandler {

	public static void print_usage_and_exit() {

		System.out.println("usage:");
		System.out.println("     boardclient server " +
			"GET count");
		System.out.println("or   boardclient server " +
			"POST uid title message");
		System.out.println("or   boardclient server " +
			"LISTEN");

		System.exit(0);
	}

	public void callback(Message msg) throws RemoteException {

		msg.print();
	}

	public static void main(String argv[]) {

		int command = 0; // 1=GET,2=POST,3=LISTEN
		String server = "";
		int count = 0;
		int uid = 0;
		String title = "";
		String message = "";

		// Parse arguments

		if (argv.length == 3 && argv[1].equals("GET")) {

			server = argv[0];

			try {
				count = Integer.parseInt(argv[2]);
			}
			catch (NumberFormatException e) {
				System.out.println("Error: " +
					"count must be an integer greater 0");
				print_usage_and_exit();
			}
			if (count <= 0) {
				System.out.println("Error: " +
					"count must be an integer greater 0");
				print_usage_and_exit();
			}

			command = 1;
		}

		else if (argv.length == 5 && argv[1].equals("POST")) {

			server = argv[0];

			try {
				uid = Integer.parseInt(argv[2]);
			}
			catch (NumberFormatException e) {
				System.out.println("Error: " +
					"UID must be an integer");
				print_usage_and_exit();
			}

			title = argv[3];
			message = argv[4];

			command = 2;
		}

		else if (argv.length == 2 && argv[1].equals("LISTEN")) {

			server = argv[0];

			command = 3;
		}

		else {
			print_usage_and_exit();
		}

		// Get Board Object
		Registry registry;
		IBoard board;
		String objName = "board";

		try {
			registry = LocateRegistry.getRegistry(server, 10412);
			board = (IBoard)registry.lookup(objName);

			// Execute command
			if (command == 1) { // GET

				Vector<Message> resVec;

				resVec = board.get(count);
				if (resVec != null && resVec.size() > 0) {
					Iterator it = resVec.iterator();

					while (it.hasNext())
						System.out.println(it.next());
				}
				else {
					System.out.println("Board empty");
				}
			}

			else if (command == 2) { // POST

				try {
					board.post(
						new Message(uid, title, message)
					);

					System.out.println("OK");
				}
				catch (RemoteException e) {
					System.out.println("Error while posting message");
				}
			}

			else if (command == 3) { // LISTEN

				ICallbackHandler cbhObj;
				ICallbackHandler cbhStub;

				cbhObj = new boardclient();
				cbhStub = (ICallbackHandler)
					UnicastRemoteObject.exportObject(
						cbhObj, 0);

				board.listen(cbhStub);
			}
		}
		catch (Exception e) {
			System.out.println(e.getMessage());
			//e.printStackTrace();
		}
	}
}

