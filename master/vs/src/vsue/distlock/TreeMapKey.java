package vsue.distlock;

import org.jgroups.Address;

public class TreeMapKey implements Comparable<TreeMapKey> {
	private int time;
	private Address addr;
	
	public TreeMapKey(int t, Address a)
	{
		time = t; 
		addr = a;
	}

	@Override
	public int compareTo(TreeMapKey o) {
		if (this.time > o.time)
			return 1;
		if (o.time > this.time)
			return -1;
		
		return this.addr.compareTo(o.addr);
	}
}
