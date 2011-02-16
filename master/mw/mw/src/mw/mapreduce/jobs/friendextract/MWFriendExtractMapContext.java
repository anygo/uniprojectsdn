package mw.mapreduce.jobs.friendextract;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;

import mw.mapreduce.core.MWMapContext;

public class MWFriendExtractMapContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> extends MWMapContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> {

	public MWFriendExtractMapContext(String inFile, long startIndex,
			long length, Comparator<KEYOUT> comp, String tmpFileName) {
		super(inFile, startIndex, length, comp, tmpFileName);
	}
	
	int c = 0;
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
		} 
		
		String id = null;
		currentVal = new ArrayList<VALUEIN>();
		
		String friends = new String();
		
		
		
		int count = 0;
		// id="next" suchen
		while (true) {
			
			if (line == null) { // Datei zu Ende
				break;
			}
			
			if (line.contains("rel=\"friend\"") && !line.contains(" alt=")) {
				System.out.println(c++);;
			}
			
			if (line.contains("id=\"next\"")) {
				if (id == null) { // id vom aktuellen profil
					String[] tmp = line.split("/");
					String roi = tmp[tmp.length-2];
					id = roi.substring(0,roi.indexOf("\""));
					currentKey = (KEYIN) id;
				} else {
					break; // naechste ID vom naechsten Profil eingelesen
				}
			} else if (id != null && line.contains("rel=\"friend\"") && !line.contains(" alt=")) {
				count++;
				String[] tmp = line.split("rel=\"friend\"");
				String roi = tmp[0];
				String val = roi.substring(roi.lastIndexOf("/")+1, roi.lastIndexOf("\""));
				if (count > 0)
					friends = friends + "~!~" + val;
				else
					friends = val;
			} else {
				// zeile ignorieren
			}
			
			
			try {
				line = textFileReader.forceReadLine();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		if (id != null) {
			currentVal.add((VALUEIN) friends);
			return true;
		}
		else {
			return false;
		}
	}
}
