package mw.mapreduce.core;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import mw.mapreduce.util.MWKeyValuesIterator;
import mw.mapreduce.util.MWTextFileReader;

public abstract class MWContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> 
	implements MWKeyValuesIterator<KEYIN, VALUEIN> {
	
	protected int currentLine;
	protected KEYIN currentKey;
	protected List<VALUEIN> currentVal;
	protected MWTextFileReader textFileReader;
	protected final String DELIM = "~!~";
	protected String outFile;
	
	public MWContext(String inFile, long startIndex, long length) {
		
		currentLine = 0;
		try {
			textFileReader = new MWTextFileReader(inFile, startIndex, length);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
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
			String[] tmp = line.split(DELIM);
			
			if(tmp.length == 1) {
				currentKey = (KEYIN) Integer.toString(currentLine);
				currentVal.add((VALUEIN) tmp[0]);
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
	
	public KEYIN getCurrentKey() {
			
		return currentKey;
	}
	
	public Iterable<VALUEIN> getCurrentValues() {

		return currentVal;
	}
	
	public abstract void write(KEYOUT key, VALUEOUT value) throws IOException;
	
	public abstract void outputComplete() throws IOException;
	
}
