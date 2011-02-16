package mw.zookeeper;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.StringTokenizer;

public class MWDataTree {
	
	protected class MWTreeNode {
		public MWStat stat;
		public byte[] data;
		public Map<String, MWTreeNode> children;
		long ephemeralOwner;
		
		public MWTreeNode(byte[] data, long time) {
			this.data = data;
			this.stat = new MWStat(time, 0);
			this.children = new HashMap<String, MWTreeNode>();
			this.ephemeralOwner = -1;
		}
	}
	
	
	protected MWTreeNode root;
	
	public MWDataTree() {
		root = new MWTreeNode(new byte[0], 0);
	}
	
	public String create(String path, byte[] data, long time) throws MWZooKeeperException {
		return create(path, data, time, -1);
	}
	
	public synchronized String create(String path, byte[] data, long time, long ephemeralOwner) throws MWZooKeeperException {
		
		StringTokenizer st = new StringTokenizer(path, "/");
		
		MWTreeNode cur = root;		
		String str = null;
		while (st.hasMoreTokens()) {
			str = st.nextToken();
			if (cur.children.containsKey(str)) {
				if (!st.hasMoreTokens()) {
					throw new MWZooKeeperException("Ordner " + str + " schon vorhanden (" + path + ")!");
				} else {
					cur = cur.children.get(str);
				}
			} else if (st.hasMoreTokens()) {
				throw new MWZooKeeperException("Ueberordner " + str + " existiert noch nicht (" + path + ")!");
			}
		}
		
		MWTreeNode node = new MWTreeNode(data, time);	
		node.ephemeralOwner = ephemeralOwner;
		cur.children.put(str, node);
		
		return path;
	}
	
	public synchronized void delete(String path, int version) throws MWZooKeeperException {
		
		StringTokenizer st = new StringTokenizer(path, "/");
		
		MWTreeNode cur = root;		
		String str = null;
		while (st.hasMoreTokens()) {
			str = st.nextToken();
			
			if (!cur.children.containsKey(str)) {
				throw new MWZooKeeperException("Fehler bei delete(" + path + ") bei Ordner " + str + "!");
			}
		
			if (st.hasMoreTokens()) {
				cur = cur.children.get(str);
			} 
		}
		
		if (cur.children.get(str).stat.version == version) {
			cur.children.remove(str);
		} else {
			throw new MWZooKeeperException("Versionsnummern stimmen nicht ueberein!");
		}
	}
	
	public synchronized MWStat setData(String path, byte[] data, int version, long time) throws MWZooKeeperException {
		
		StringTokenizer st = new StringTokenizer(path, "/");
		
		MWTreeNode cur = root;		
		String str = null;
		while (st.hasMoreTokens()) {
			str = st.nextToken();
			
			if (!cur.children.containsKey(str)) {
				throw new MWZooKeeperException("Pfad " + path + " existiert nicht (" + str + ")!");
			}

			cur = cur.children.get(str);
		}

		if (cur.stat.version == version) {
			cur.data = data;
			cur.stat.version++;
			cur.stat.time = time;
		} else {
			throw new MWZooKeeperException("Versionsnummern stimmen nicht ueberein!");
		}
		
		return cur.stat;
	}
	
	public synchronized byte[] getData(String path, MWStat stat) throws MWZooKeeperException {
		
		StringTokenizer st = new StringTokenizer(path, "/");
		
		MWTreeNode cur = root;		
		String str = null;
		while (st.hasMoreTokens()) {
			str = st.nextToken();
			
			if (!cur.children.containsKey(str)) {
				throw new MWZooKeeperException("Pfad " + path + " existiert nicht (" + str + ")!");
			}

			cur = cur.children.get(str);
		}
		
		stat.version = cur.stat.version;
		stat.time = cur.stat.time;
		return cur.data;
	}
	
	public synchronized void killSession(long ephemeralOwner) {
		
		traverseTree(root, ephemeralOwner);
	}
	
	protected void traverseTree(MWTreeNode cur, long ephemeralOwner) {
		
		if (cur.children.isEmpty()) {
			return;
		}
		
		for (Entry<String, MWTreeNode> entry: cur.children.entrySet()) {
			if (entry.getValue().ephemeralOwner == ephemeralOwner) {
				cur.children.remove(entry.getKey());
				System.out.println("Deleted: " + entry.getKey());
			} else {
				traverseTree(entry.getValue(), ephemeralOwner);
			}
		}
		
	}
}
