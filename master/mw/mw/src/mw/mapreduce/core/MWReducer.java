package mw.mapreduce.core;

import java.io.IOException;

public class MWReducer<KEYIN, VALUEIN, KEYOUT, VALUEOUT> implements Runnable {

	MWReduceContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> context;
	
	@Override
	public void run() {
		
		while (context.nextKeyValues()) {
			reduce(context.getCurrentKey(), context.getCurrentValues(), context);
		}
		try {
			context.outputComplete();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void setContext(MWReduceContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> context) {
		
		this.context = context;
	}
	
	@SuppressWarnings("unchecked")
	protected void reduce(KEYIN key, Iterable<VALUEIN> values, MWReduceContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> context) {
		
		for (VALUEIN val: values) {
			try {
				context.write((KEYOUT) key, (VALUEOUT) val);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
