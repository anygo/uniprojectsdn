package mw.zookeeper;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import org.apache.zookeeper.zab.MultiZab;
import org.apache.zookeeper.zab.Zab;
import org.apache.zookeeper.zab.ZabCallback;
import org.apache.zookeeper.zab.ZabStatus;
import org.apache.zookeeper.zab.ZabTxnCookie;

public class MWZooKeeperServer implements ZabCallback {
	
	public MWDataTree tree;
	public Map<Integer, ObjectOutputStream> outputstreams;
	
	public MWZooKeeperServer() {
		tree = new MWDataTree();
		outputstreams = new HashMap<Integer, ObjectOutputStream>();
	}
	
	public Serializable deserialize(byte[] arr) throws Exception {
		
		ByteArrayInputStream bais = new ByteArrayInputStream(arr);
		ObjectInputStream ois = new ObjectInputStream(bais);
		Serializable ret = (Serializable)ois.readObject();
		bais.close();
		ois.close();
		
		return ret;
	}
	
	public class ZooWorkerThread implements Runnable {

		private ObjectOutputStream out;
		private ObjectInputStream in;
		private String clientName;
		private MWDataTree tree;
		private Zab zabNode;
		
		ZooWorkerThread(Socket client, ObjectInputStream in, ObjectOutputStream out, MWDataTree tree, Zab zabNode) {	
			this.in = in;
			this.out = out;
			this.tree = tree;
			this.clientName = client.getInetAddress().getCanonicalHostName();
			this.zabNode = zabNode;
		}
		
		public byte[] serialize(Serializable obj) throws Exception {
			
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			ObjectOutputStream oos = new ObjectOutputStream(baos);
			oos.writeObject(obj);
			oos.close();
			baos.close();
			return baos.toByteArray();
		}

		@Override
		public void run() {
		
			int hash = out.hashCode();
			
			while (true) {
				MWZooKeeperMessage msg = null;
				try {
					msg = (MWZooKeeperMessage) in.readObject();
				} catch (Exception e) {
					System.err.println(e.getMessage());
					System.out.println("### Verbindung zu " + clientName + " ist abgebrochen");
					msg = new MWZooKeeperMessage("kill");
					msg.outputStreamHash = hash;
					try {
						zabNode.propose(serialize(msg));
					} catch (Exception e1) {
						System.err.println(e1.getMessage());
					}
					return;
				}

				System.out.println(clientName + ": " + msg.command + " " + msg.path);
				
				msg.outputStreamHash = hash;
				
				if (msg.command.equalsIgnoreCase("create")) {
	
					try {
						zabNode.propose(serialize(msg));
					} catch (Exception e) {
						e.printStackTrace();
					}

				} else if (msg.command.equalsIgnoreCase("delete")) {
					
					try {
						zabNode.propose(serialize(msg));
					} catch (Exception e) {
						e.printStackTrace();
					}
					
				} else if (msg.command.equalsIgnoreCase("setData")) {
					
					try {
						zabNode.propose(serialize(msg));
					} catch (Exception e) {
						e.printStackTrace();
					}

				} else if (msg.command.equalsIgnoreCase("getData")) {
					
					byte[] data = null;
					MWStat stat = new MWStat();
					
					try {
						data = tree.getData(msg.path, stat);
					} catch (MWZooKeeperException e) {
						msg.exception = e;
						System.err.println(e.getMessage());
					}
					
					MWZooKeeperMessage msgOut = new MWZooKeeperMessage("getDataReturnMessage");
					msgOut.data = data;
					msgOut.stat = stat;	
					msgOut.exception = msg.exception;
					
					try {
						out.writeObject(msgOut);
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}		
		}
	}
	
	@Override
	public void deliver(ZabTxnCookie arg0, byte[] arg1) {
		
		MWZooKeeperMessage msg = null;
		try {
			msg = (MWZooKeeperMessage) deserialize(arg1);
		} catch (Exception e) {
			e.printStackTrace();
			return;
		}

		if (msg.command.equalsIgnoreCase("create")) {
		
			try {
				tree.create(msg.path, msg.data, msg.stat.time, msg.ephemeralNode ? msg.outputStreamHash : -1);
			} catch (MWZooKeeperException e) {
				System.err.println(e.getMessage());
				msg.exception = e;
			} 
		} else if (msg.command.equalsIgnoreCase("delete")) {
			
			try {
				tree.delete(msg.path, msg.stat.version);
			} catch (MWZooKeeperException e) {
				System.err.println(e.getMessage());
				msg.exception = e;
			}
		} else if (msg.command.equalsIgnoreCase("setData")) {
			
			MWStat stat = new MWStat();
			try {
				stat = tree.setData(msg.path, msg.data, msg.stat.version, msg.stat.time);
			} catch (MWZooKeeperException e) {
				System.err.println(e.getMessage());
				msg.exception = e;
			}
			
			msg.stat.time = stat.time;
			msg.stat.version = stat.version;
			
		} else if (msg.command.equalsIgnoreCase("kill")) {
			
			tree.killSession(msg.outputStreamHash);
			if (outputstreams.containsKey(msg.outputStreamHash))
				outputstreams.remove(msg.outputStreamHash);
			
			return;
			
		} else {
			System.out.println("Sollte nicht vorkommen!");
		}
		
		if (outputstreams.containsKey(msg.outputStreamHash)) {
			try {
				outputstreams.get(msg.outputStreamHash).writeObject(msg);
				outputstreams.get(msg.outputStreamHash).flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
		} 
	}


	@Override
	public void deliverSync(ZabTxnCookie arg0) {
		//System.out.println("deliverSync");
		
	}


	@Override
	public void getState(OutputStream arg0) throws IOException {
		//System.out.println("getState: " + arg0);
		
	}


	@Override
	public void setState(InputStream arg0, ZabTxnCookie arg1) throws IOException {
		//System.out.println("setState " + arg0);
		
	}


	@Override
	public void status(ZabStatus arg0, String arg1) {
		//System.out.println("status: " + arg1);
		
	}
	
	
	public static void main(String[] args) throws Exception {

		ServerSocket server = new ServerSocket(Integer.parseInt(args[1]));
		
		Properties zabProperties = new Properties();
		zabProperties.setProperty("myid", args[0]);
		
		for (int i = 2; i < args.length; i++) {
			zabProperties.setProperty("peer"+Integer.toString(i-1), args[i]);
		}
		
		MWZooKeeperServer zooKeeper = new MWZooKeeperServer();
		Zab zabNode = new MultiZab(zooKeeper, zabProperties);
		
		zabNode.startup();

		
		System.out.println("### MWZooKeeperServer laeuft");

		while (true) {
			Socket client = server.accept();
			
			ObjectInputStream in = null;
			ObjectOutputStream out = null;
			
			try {
				in = new ObjectInputStream(client.getInputStream());
				out = new ObjectOutputStream(client.getOutputStream());
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
			
			zooKeeper.outputstreams.put(out.hashCode(), out);
			ZooWorkerThread worker = zooKeeper.new ZooWorkerThread(client, in, out, zooKeeper.tree, zabNode);
			Thread runner = new Thread(worker);
			runner.start();
			System.out.println("### Neuer Client angenommen");
		}
	}
}
