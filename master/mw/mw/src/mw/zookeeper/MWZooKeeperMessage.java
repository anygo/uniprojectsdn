package mw.zookeeper;

import java.io.Serializable;

public class MWZooKeeperMessage implements Serializable {

	private static final long serialVersionUID = -6181292557718958696L;
	
	public String command;
	public byte[] data;
	public MWStat stat;
	public String path;
	public MWZooKeeperException exception;
	public int outputStreamHash;
	public boolean ephemeralNode;
	
	public MWZooKeeperMessage(String command) {
		this.command = command;
		this.data = new byte[0];
		this.stat = new MWStat();
		this.path = new String();
		this.exception = null;
		this.outputStreamHash = 0;
		this.ephemeralNode = false;
	}
	
}
