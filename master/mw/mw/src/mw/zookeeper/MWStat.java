package mw.zookeeper;

import java.io.Serializable;


public class MWStat implements Serializable {

	private static final long serialVersionUID = 7031186626025026665L;
	
	public long time;
	public int version;
	
	public MWStat() {
		this.time = -1;
		this.version = -1;
	}
	
	public MWStat(long time, int version) {
		this.time = time;
		this.version = version;
	}
	


}
