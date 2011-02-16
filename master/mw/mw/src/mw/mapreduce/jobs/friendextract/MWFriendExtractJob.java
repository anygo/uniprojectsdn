package mw.mapreduce.jobs.friendextract;

import java.io.File;
import java.util.Comparator;

import mw.mapreduce.core.MWJob;
import mw.mapreduce.core.MWMapContext;
import mw.mapreduce.core.MWMapper;
import mw.mapreduce.core.MWReduceContext;
import mw.mapreduce.core.MWReducer;
import mw.mapreduce.jobs.friendcount.MWFriendCountStringComparator;

public class MWFriendExtractJob implements MWJob {

	@SuppressWarnings({ "unchecked", "rawtypes" })
	@Override
	public MWMapper createMapper(String inFile, int nMappers, int current, String tmpFileName) {
		
		File f = new File(inFile);
		if(!f.exists()) { 
			System.err.println(inFile + " Input file not found!"); 
			return null; 
		}
		long fileSize = f.length();
		
		long stepSize = fileSize / nMappers;
		long length = stepSize;
		if (nMappers == current+1) {
			length = fileSize - (nMappers-1)*stepSize;
		} 
		
		MWMapContext<String, String, String, String> context = 
			new MWFriendExtractMapContext<String, String, String, String>(inFile, current*stepSize, length, getComparator(), tmpFileName);
		
		
		MWMapper<String, String, String, String> m = new MWFriendExtractMapper<String, String, String, String>();
		m.setContext(context);
		return m;
	}

	@SuppressWarnings({ "rawtypes" })
	@Override
	public MWReducer createReducer(String inFile, int nReducers, int current, String outFileName) {
		
		File f = new File(inFile);
		if(!f.exists()) { 
			System.err.println(inFile + " Input file not found!"); 
			return null; 
		}
		long fileSize = f.length();
		
		long stepSize = fileSize / nReducers;
		long length = stepSize;
		if (nReducers == current+1) {
			length = fileSize - (nReducers-1)*stepSize;
		} 
		
		MWReduceContext<String, String, String, String> context = 
			new MWReduceContext<String, String, String, String>(inFile, current*stepSize, length, outFileName);
		
		MWReducer<String, String, String, String> m = new MWReducer<String, String, String, String>();
		m.setContext(context);
		return m;
	}

	@SuppressWarnings("rawtypes")
	@Override
	public Comparator getComparator() {
		return new MWFriendCountStringComparator<String>();
	}

	
}
