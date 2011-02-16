package mw.mapreduce.jobs.friendcount;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;

import mw.mapreduce.core.MWMapContext;

public class MWFriendCountMapContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> extends MWMapContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> {

	public MWFriendCountMapContext(String inFile, long startIndex, long length,
			Comparator<KEYOUT> comp, String tmpFileName) {
		super(inFile, startIndex, length, comp, tmpFileName);
		// TODO Auto-generated constructor stub
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public boolean nextKeyValues() {

		String line = null;
		
		try {
			line = textFileReader.readLine();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		if (line == null) {
			
			return false;
			
		} else {
			
			currentVal = new ArrayList<VALUEIN>();
			String[] tmp = line.split("\t");
			
			if(tmp.length == 1) {
				System.err.println("Error in friendCountMap");
			} else {
				currentKey = (KEYIN) tmp[0];
				for (int i = 1; i < tmp.length; i++) {
					currentVal.add((VALUEIN) tmp[i]);
				}
			}
			currentLine++;
			return true;
		}
		
	}
}
