package mw.mapreduce.jobs.friendcount;

import java.text.Collator;
import java.util.Comparator;

public class MWFriendCountComparator implements
		Comparator<MWFriendcountKeyListValuePair> {

	@Override
	public int compare(MWFriendcountKeyListValuePair o1, MWFriendcountKeyListValuePair o2) {
		
		return Collator.getInstance().compare(o1.key, o2.key);
	}

}
