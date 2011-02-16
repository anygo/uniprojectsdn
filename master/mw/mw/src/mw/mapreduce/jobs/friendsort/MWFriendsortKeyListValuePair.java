package mw.mapreduce.jobs.friendsort;

import java.util.ArrayList;
import java.util.List;

public class MWFriendsortKeyListValuePair {

	public int key;
	public int fileID;
	public List<String> values;
	
	public MWFriendsortKeyListValuePair(int key, int fileID, List<String> values) {
		
		this.key = key;
		this.fileID = fileID;
		this.values = values;
	}
	
	public MWFriendsortKeyListValuePair(int fileID, String line) {
		
		this.fileID = fileID;
		
		String[] tmp = line.split("~!~");
		values = new ArrayList<String>();
		
		key = Integer.parseInt(tmp[0]);
		
		for(int i = 1; i < tmp.length; i++)
			values.add(tmp[i]);
	}
}
