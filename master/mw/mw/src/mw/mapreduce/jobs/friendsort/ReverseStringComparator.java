package mw.mapreduce.jobs.friendsort;

import java.text.Collator;
import java.util.Comparator;

@SuppressWarnings("hiding")
public class ReverseStringComparator<String> implements Comparator<String> {

	@Override
	public int compare(String o1, String o2) {
		
		return -Collator.getInstance().compare(o1, o2);
	}

}
