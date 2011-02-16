package mw.mapreduce.jobs.friendcount;


import java.text.Collator;
import java.util.Comparator;

@SuppressWarnings("hiding")
public class MWFriendCountStringComparator<String> implements Comparator<String> {

	Collator col = Collator.getInstance();
	
	@Override
	public int compare(String o1, String o2) {
		
		return col.compare(o1, o2);
	}

}
