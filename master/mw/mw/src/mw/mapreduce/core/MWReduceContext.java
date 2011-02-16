package mw.mapreduce.core;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;


public class MWReduceContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> extends MWContext<KEYIN, VALUEIN, KEYOUT, VALUEOUT> {
	
	BufferedWriter b;
	
	public MWReduceContext(String inFile, long startIndex, long length, String outFileName) {
		
		super(inFile, startIndex, length);
		outFile = outFileName;
		try {
			b = new BufferedWriter(new FileWriter(outFile));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public void write(KEYOUT key, VALUEOUT value) throws IOException {
		
		b.write(key + DELIM + value + "\n");
	}

	@Override
	public void outputComplete() throws IOException {

		b.close();
	}

}
