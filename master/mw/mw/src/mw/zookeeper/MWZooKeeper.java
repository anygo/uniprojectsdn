package mw.zookeeper;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.util.StringTokenizer;


public class MWZooKeeper {
	
	private Socket server;
	private ObjectInputStream in;
	private ObjectOutputStream out;
	
	public MWZooKeeper() {
		
		try {
			server = new Socket("faui07f.informatik.uni-erlangen.de", 11234);
		} catch (Exception e) {
			e.printStackTrace();
		}
			
		try {
			out = new ObjectOutputStream(server.getOutputStream());
			in = new ObjectInputStream(server.getInputStream());
		} catch (IOException e) {
			e.printStackTrace();
		}		
		
	}

	public String create(String path, byte[] data, long time, boolean ephemeralNode) throws MWZooKeeperException {
		
		MWZooKeeperMessage msg = new MWZooKeeperMessage("create");
		msg.data = data;
		msg.stat = new MWStat(time, 0);
		msg.path = path;
		msg.ephemeralNode = ephemeralNode;
		
		MWZooKeeperMessage answer = null;
		try {
			out.writeObject(msg);
			answer = (MWZooKeeperMessage) in.readObject();
		} catch (Exception e) {
			e.printStackTrace();
		} 
		
		if (answer.exception != null) {
			throw answer.exception;
		} 

		return answer.path;
	}
	
	public void delete(String path, int version) throws MWZooKeeperException {
		
		MWZooKeeperMessage msg = new MWZooKeeperMessage("delete");
		msg.stat = new MWStat(-1, version);
		msg.path = path;

		MWZooKeeperMessage answer = null;
		try {
			out.writeObject(msg);
			answer = (MWZooKeeperMessage) in.readObject();
		} catch (Exception e) {
			e.printStackTrace();
		} 
		
		if (answer.exception != null) {
			throw answer.exception;
		} 
		
	}
	
	public MWStat setData(String path, byte[] data, int version, long time) throws MWZooKeeperException {
		
		MWZooKeeperMessage msg = new MWZooKeeperMessage("setData");
		msg.data = data;
		msg.stat = new MWStat(time, 0);
		msg.stat.version = version;
		msg.stat.time = time;
		msg.path = path;
		
		MWZooKeeperMessage answer = null;
		try {
			out.writeObject(msg);
			answer = (MWZooKeeperMessage) in.readObject();
		} catch (Exception e) {
			e.printStackTrace();
		} 
		
		if (answer.exception != null) {
			throw answer.exception;
		} 

		return answer.stat;
	}
	
	public byte[] getData(String path, MWStat stat) throws MWZooKeeperException {
		
		MWZooKeeperMessage msg = new MWZooKeeperMessage("getData");
		msg.path = path;
		
		MWZooKeeperMessage msgIn = null;
		
		try {
			out.writeObject(msg);
			msgIn = (MWZooKeeperMessage) in.readObject();
		} catch (Exception e) {
			e.printStackTrace();
		} 

		if (msgIn.exception != null) {
			throw msgIn.exception;
		} 
		
		stat.version = msgIn.stat.version;
		stat.time = msgIn.stat.time;
	
		return msgIn.data;
	}
	
	
	public static void main(String[] args) throws Exception {

		MWZooKeeper zookeeper = new MWZooKeeper();
		
		
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		
		while (true) {
			
			try {
				String cmd = br.readLine();
				StringTokenizer st = new StringTokenizer(cmd);
				
				String[] params = new String[st.countTokens() - 1];
				String fct = st.nextToken();
		
				for (int i = 0; i < params.length; i++) {
					params[i] = st.nextToken();
				}
				
				if (fct.equalsIgnoreCase("create")) {
					String answ = zookeeper.create(params[0], params[1].getBytes(), System.currentTimeMillis(), false);
					System.out.println(answ);
				} else if (fct.equalsIgnoreCase("createE")) {
					String answ = zookeeper.create(params[0], params[1].getBytes(), System.currentTimeMillis(), true);
					System.out.println(answ + " (ephemeral)");
				} else if (fct.equalsIgnoreCase("delete")) {
					zookeeper.delete(params[0], Integer.parseInt(params[1]));
				} else if (fct.equalsIgnoreCase("getData")) {
					MWStat answStat = new MWStat();
					byte[] answByte = zookeeper.getData(params[0], answStat);
					System.out.println("Antwort: " + new String(answByte) + " - Version: " + answStat.version);
				} else if (fct.equalsIgnoreCase("setData")) {
					MWStat newAnswStat = zookeeper.setData(params[0], params[1].getBytes(), Integer.parseInt(params[2]), System.currentTimeMillis());
					System.out.println("Neue Version: " + newAnswStat.version);
				} else if (fct.equalsIgnoreCase("attack")) {
					for (int i = 0; i < 1000; i++) {
						try {
						MWStat answStat = new MWStat();
						zookeeper.getData("/attack", answStat);
						answStat = zookeeper.setData("/attack", new byte[] {1}, answStat.version, System.currentTimeMillis());
						System.out.println("Neue Version: " + answStat.version);
						} catch (MWZooKeeperException e) {
							System.err.println(e.getMessage());
						}
					}
				} else if (fct.equalsIgnoreCase("readtest")) {
					int n = 50;
					MWStat answStat = new MWStat();
					long start = System.currentTimeMillis();
					for (int i = 0; i < n; i++) {
						zookeeper.getData("/attack", answStat);
					}
					long end = System.currentTimeMillis();
					
					System.out.println("Read Average time: " + (end-start)/n + "ms");
				} else if (fct.equalsIgnoreCase("writetest")) {
					int n = 50;
					MWStat answStat = new MWStat();
					zookeeper.getData("/attack", answStat);
					long start = System.currentTimeMillis();
					for (int i = 0; i < n; i++) {
						answStat = zookeeper.setData("/attack", new byte[] {1}, answStat.version, System.currentTimeMillis());
					}
					long end = System.currentTimeMillis();
					
					System.out.println("Write Average time: " + (end-start)/n + "ms");
				}else {
					System.err.println("Invalid command");
				}
				
			} catch (Exception e) {
				
				if (e instanceof ArrayIndexOutOfBoundsException) {
					System.err.println("Provide all required parameters, stupid!");
				} else {
					if(!e.getMessage().isEmpty())
					System.err.println(e.getMessage());
				}
			}
		}
	}
}
