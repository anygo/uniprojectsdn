package vsue.rmi;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;

public class VSBoardRMIClient implements VSBoardListener {

	public static void main(String args[]) throws Exception {

		// Registry erzeugen und "Verbindung" herstellen
		Registry registry = LocateRegistry.getRegistry("faui06n", 12345);
		VSBoard board = (VSBoard) registry.lookup("board");

		// Buffered Reader anlegen
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

		System.out.println("Connected... post your messages...");

		while (true) {
			System.out.print("$> ");

			String str = br.readLine();

			if (str.equalsIgnoreCase("initTest")) {
				for (int i = 0; i < 10000; ++i) {
					VSBoardMessage message = new VSBoardMessage();
					message.setMessage("I am a message.");
					message.setTitle("I am a title.");
					message.setUid((int) (Math.random() * 100000));

					board.post(message);

					if (i % 100 == 0)
						System.out.println(i);
				}
				System.out.println("init'ed");
				continue;
			} else if (str.equalsIgnoreCase("runTest")) {

				final int N_TIMES = 20;

				BufferedWriter bw = new BufferedWriter(new FileWriter(
						"java_rmi.txt"));

				for (int j = 0; j <= 10000; j += (j < 1000) ? 10
						: (j < 10000) ? 100 : 1000) {
					long start = System.currentTimeMillis();
					for (int i = 0; i < N_TIMES; ++i)
						board.get(j);
					long end = System.currentTimeMillis();
					long avgTime = (end - start) / N_TIMES;

					bw.write(j + " " + avgTime + "\n");

					System.out.println("avg for get(" + j + "): " + avgTime + "ms");
					bw.flush();
				}

				bw.close();

				continue;
			}

			String[] parts = str.split("\"");

			if (parts.length == 1) {
				parts = parts[0].split(" ");
				if (parts.length == 1 && parts[0].equals("listen")) // listen
				{
					VSBoardListener vsbl = new VSBoardRMIClient();
					VSBoardListener vsblStub = (VSBoardListener) UnicastRemoteObject
							.exportObject(vsbl, 0);
					board.listen(vsblStub);

					System.out.println("listening...");
				} else if (parts.length == 2
						&& parts[0].equalsIgnoreCase("get")) // get
				{
					int n = Integer.parseInt(parts[1]);
					VSBoardMessage[] messages = board.get(n);

					for (VSBoardMessage vsbm : messages) {
						System.out.println("\t-> " + vsbm);
					}
				} else {
					System.out.println("error.. try again!");
					continue;
				}

			} else if (parts.length == 4) // post
			{
				String[] leftParts = parts[0].split(" ");
				String method = leftParts[0];

				if (leftParts.length != 2 || !method.equalsIgnoreCase("post")) {
					System.out.println("error.. try again!");
					continue;
				}

				int uid = Integer.parseInt(leftParts[1]);
				String title = parts[1];
				String msg = parts[3];

				VSBoardMessage message = new VSBoardMessage();
				message.setMessage(msg);
				message.setTitle(title);
				message.setUid(uid);

				board.post(message);
			} else {
				System.out.println("error.. try again!");
				continue;
			}

		}
	}

	@Override
	public void newMessage(VSBoardMessage message) throws RemoteException {
		System.out.println("Listener - new Message: " + message);
	}
}
