package mw.mapreduce.jobs.friendsort;

import java.io.IOException;

import mw.mapreduce.core.MWMapContext;
import mw.mapreduce.core.MWMapper;

public class MWFriendSortMapper<KEYIN, VALUEIN, KEYOUT, VALUEOUT> extends MWMapper<KEYIN, VALUEIN, KEYOUT, VALUEOUT> {

	@SuppressWarnings("unchecked")
	@Override
	protected void map(KEYIN key, VALUEIN value, MWMapContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> context) {
		
		String v = (String)value;
		String[] vals = v.split("\t");
		
		try {
			context.write((KEYOUT) vals[1], (VALUEOUT) vals[0]);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}
