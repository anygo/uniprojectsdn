package mw.mapreduce.jobs.friendsort;

import java.util.Comparator;

public class MWFriendSortComparator implements Comparator<MWFriendsortKeyListValuePair> {

	@Override
	public int compare(MWFriendsortKeyListValuePair o1, MWFriendsortKeyListValuePair o2) {
		
		return - (o1.key - o2.key);
	}

}
