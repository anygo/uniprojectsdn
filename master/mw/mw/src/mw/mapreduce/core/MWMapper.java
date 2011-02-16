package mw.mapreduce.core;

import java.io.IOException;

public class MWMapper<KEYIN, VALUEIN, KEYOUT, VALUEOUT> implements Runnable {

	MWMapContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> context;
	
	@Override
	public void run() {
		
		while (context.nextKeyValues()) {
			map(context.getCurrentKey(), context.getCurrentValues().iterator().next(), context);
		}
		try {
			context.outputComplete();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void setContext(MWMapContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> context) {
		
		this.context = context;
	}
	
	@SuppressWarnings("unchecked")
	protected void map(KEYIN key, VALUEIN value, MWMapContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> context) {
		
		try {
			context.write((KEYOUT) key, (VALUEOUT) value);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
