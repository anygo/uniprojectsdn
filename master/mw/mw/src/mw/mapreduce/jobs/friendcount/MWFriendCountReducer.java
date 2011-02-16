package mw.mapreduce.jobs.friendcount;

import java.io.IOException;

import mw.mapreduce.core.MWReduceContext;
import mw.mapreduce.core.MWReducer;

public class MWFriendCountReducer<KEYIN, VALUEIN, KEYOUT, VALUEOUT> extends MWReducer<KEYIN, VALUEIN, KEYOUT, VALUEOUT> {
	
	@SuppressWarnings("unchecked")
	@Override
	protected void reduce(KEYIN key, Iterable<VALUEIN> values, MWReduceContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> context) {
		
		String name = null;
		int count = 0;
		
		for (VALUEIN val: values) {
			String v = (String) val;
			if (v.contains("|")) {
				name = v.substring(1);
			}
			else 
				count++;
		}

		
		try {
			context.write((KEYOUT) name, (VALUEOUT) Integer.toString(count));
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
	}
	
}
