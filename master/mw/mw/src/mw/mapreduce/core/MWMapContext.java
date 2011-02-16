package mw.mapreduce.core;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;


public class MWMapContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> extends MWContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> {

	protected Map<KEYOUT, List<VALUEOUT>> storage;
	
	public MWMapContext(String inFile, long startIndex, long length, Comparator<KEYOUT> comp, String tmpFileName) {
		
		super(inFile, startIndex, length);
		storage = new TreeMap<KEYOUT, List<VALUEOUT>>(comp);
		outFile = tmpFileName;
	}

	@Override
	public void write(KEYOUT key, VALUEOUT value) throws IOException {
		
		if(storage.containsKey(key)) {
			storage.get(key).add(value);
		} else {
			List<VALUEOUT> l = new ArrayList<VALUEOUT>();
			l.add(value);
			storage.put(key, l);	
		}
	}

	@Override
	public void outputComplete() throws IOException {
		
		Set<KEYOUT> allKeys = storage.keySet();
		
		BufferedWriter b = new BufferedWriter(new FileWriter(outFile)); 
		for(KEYOUT k: allKeys) {
			List<VALUEOUT> valList = storage.get(k);
			String valString = new String();
			for (VALUEOUT v: valList) {
				valString = valString + DELIM + v;
			}
			b.write((String) k + valString + "\n");
		}
		b.close();
	}
	

}
