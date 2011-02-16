package mw.mapreduce.jobs.friendcount;

import java.util.ArrayList;
import java.util.List;

public class MWFriendcountKeyListValuePair {

	public String key;
	public int fileID;
	public List<String> values;
	
	
	public MWFriendcountKeyListValuePair(int fileID, String line) {
		
		this.fileID = fileID;
		
		String[] tmp = line.split("~!~");
		values = new ArrayList<String>();
		
		key = tmp[0];
		
		for(int i = 1; i < tmp.length; i++)
			values.add(tmp[i]);
	}
}
