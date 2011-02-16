package mw.mapreduce.core;

import java.util.Comparator;

public interface MWJob {
	
	@SuppressWarnings("rawtypes")
	public MWMapper createMapper(String inFile, int nMappers, int current, String tmpFileName);
	@SuppressWarnings("rawtypes")
	public MWReducer createReducer(String inFile, int nReducers, int current, String outFileName);
	@SuppressWarnings("rawtypes")
	public Comparator getComparator();
}
