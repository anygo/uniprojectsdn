package mw.mapreduce.jobs.friendextract;

import java.io.IOException;

import mw.mapreduce.core.MWMapContext;
import mw.mapreduce.core.MWMapper;

public class MWFriendExtractMapper<KEYIN, VALUEIN, KEYOUT, VALUEOUT> extends MWMapper<KEYIN, VALUEIN, KEYOUT, VALUEOUT> {

	@Override
	@SuppressWarnings("unchecked")
	protected void map(KEYIN key, VALUEIN value, MWMapContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> context) {

		String[] tmp = ((String)value).split("~!~");
		
		for (int i = 0; i < tmp.length; i++) {
			try {
				// idA - idB
				// idB - idA
				context.write((KEYOUT) key, (VALUEOUT) tmp[i]);
				context.write((KEYOUT) tmp[i], (VALUEOUT) key);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
