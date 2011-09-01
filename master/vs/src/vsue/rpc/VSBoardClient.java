package vsue.rpc;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

import vsue.VSConstants;
import vsue.rmi.VSBoard;
import vsue.rmi.VSBoardListener;
import vsue.rmi.VSBoardMessage;

public class VSBoardClient implements VSBoardListener {

	public VSBoardClient() {
	}

	public void startShell() {
		Registry registry = null;
		VSBoard board = null;

		try {
			registry = LocateRegistry.getRegistry(
					VSConstants.VSBOARD_SERVER_NAME,
					VSConstants.REGISTRY_PORT_BOARD_SERVER);
			board = (VSBoard) registry.lookup("board");

			System.out.println("Connected, post your messages...");
			while (true) {
				try {
					parseLine(board);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		} catch (Exception e) {
			System.err.println("Exception while starting client");
			e.printStackTrace();
		}
	}

	private void parseLine(VSBoard board) throws Exception {
		BufferedReader br = null;

		System.out.print("$> ");

		br = new BufferedReader(new InputStreamReader(System.in));
		String str = br.readLine();

		if (str.equalsIgnoreCase("initTest")) {
			for (int i = 0; i < 100000; ++i) {
				VSBoardMessage message = new VSBoardMessage();
				message.setMessage("I am a message.");
				message.setTitle("I am a title.");
				message.setUid((int) (Math.random() * 1000000));

				board.post(message);

				if (i % 100 == 0)
					System.out.println(i);
			}
			System.out.println("init'ed");
			return;
		} else if (str.equalsIgnoreCase("runTest")) {

			final int N_TIMES = 20;

			BufferedWriter bw = new BufferedWriter(new FileWriter(
					"hart_codiert.txt"));

			for (int j = 0; j <= 10000; j += (j < 1000) ? 10
					: (j < 10000) ? 100 : 1000) {
				long start = System.currentTimeMillis();
				for (int i = 0; i < N_TIMES; ++i)
					board.get(j);
				long end = System.currentTimeMillis();
				long avgTime = (long) ((end - start) / new Float(N_TIMES));

				bw.write(j + " " + avgTime + "\n");

				System.out.println("avg for get(" + j + "): " + avgTime + "ms");
				bw.flush();
			}
			bw.close();
			return;
		}

		String[] parts = str.split("\"");

		if (parts.length == 1) {
			parts = parts[0].split(" ");
			if (parts.length == 1 && parts[0].equals("listen")) {
				//VSBoardListener vsblStub = (VSBoardListener) VSRemoteObjectManager
				//		.getInstance().exportObject(this);
				board.listen(this);
				System.out.println("listening...");
			} else if (parts.length == 2 && parts[0].equalsIgnoreCase("get")) {
				int n = Integer.parseInt(parts[1]);
				VSBoardMessage[] messages = board.get(n);

				for (VSBoardMessage vsbm : messages) {
					System.out.println("\t-> " + vsbm);
				}
			} else {
				System.out.println("syntax error... try again!");
				return;
			}
		} else if (parts.length == 4) { // post
			String[] leftParts = parts[0].split(" ");
			String method = leftParts[0];

			if (leftParts.length != 2 || !method.equalsIgnoreCase("post")) {
				System.out.println("syntax error... try again!");
				return;
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
			System.out.println("syntax error... try again!");
		}
	}

	@Override
	public void newMessage(VSBoardMessage message) throws RemoteException {
		System.out.println("Listener - new Message: " + message);
	}

	public static void main(String args[]) {
		VSBoardClient client = new VSBoardClient();
		client.startShell();
	}
}
