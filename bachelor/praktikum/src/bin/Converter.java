package bin;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import io.*;
import statistics.Sample;

public class Converter {
	public static final short LABEL_SIZE = 12;
	
	public static final String SYNOPSIS = 
		"Translate between various file formats.\n\n" +
		"usage: java bin.Converter in_format out_format < data_in > data_out\n\n" +
		"formats:\n" +
		"  ufv,dim\n" +
		"    Unlabeled feature data, 4 byte (float) per sample dimension\n" +
		"  lfv,dim,label1,label2,...,labeln\n" +
		"    Labeled feature data; 12 byte label, then 4 byte (float) per sample.\n" +
		"    Label ID will be attached according to the sequence of labels in the\n" +
		"    argument: label1 -> 0, label2 -> 1, etc. as the Sample class requires\n" +
		"    numeric labels.\n" +
		"  frame\n" +
		"    Unlabeled feature data, 8 byte (double) per sample dimension\n" +
		"  sample\n" +
		"    Labeled feature data using the statistics.Sample class\n" +
		"  ascii\n" +
		"    Unlabeled ASCII data: TAB separated double values, one sample per line.\n" +
		"  ascii_label\n" +
		"    Labelled ASCII data: TAB separated values, first field is label.\n";
	
	public static enum Format {
		UFV,
		LFV,
		FRAME,
		SAMPLE,
		ASCII,
		ASCII_L
	}
	
	public static int fd = 24;
	
	public static HashMap<String, Integer> lookup1 = new HashMap<String, Integer>();
	public static HashMap<Integer, String> lookup2 = new HashMap<Integer, String>();
	
	private static int lab = 1;
	
	/**
	 * Analyze the format string
	 */
	public static Format determineFormat(String arg) {
		if (arg.startsWith("ufv")) {
			fd = Integer.parseInt(arg.substring(4));
			return Format.UFV;
		} else if (arg.startsWith("lfv")) {
			String [] list = arg.split(",");
			fd = Integer.parseInt(list[1]);
			
			for (int i = 2; i < list.length; ++i) {
				lookup1.put(list[i], lab);
				lookup2.put(lab, list[i]);
				lab++;
			}
			return Format.LFV;
		} else if (arg.equals("frame"))
			return Format.FRAME;
		else if (arg.equals("sample"))
			return Format.SAMPLE;
		else if (arg.equals("ascii"))
			return Format.ASCII;
		else if (arg.equals("ascii_label"))
			return Format.ASCII_L;
		else
			throw new RuntimeException("invalid format \"" + arg + "\"");
	}
	
	public static void main(String[] args) throws Exception {
		if (args.length != 2) {
			System.err.println(SYNOPSIS);
			System.exit(1);
		}
		
		// get the formats
		Format inFormat = determineFormat(args[0]);
		Format outFormat = determineFormat(args[1]);
		
		// possible readers
		ObjectInputStream ois = null;
		BufferedReader br = null;
		FrameReader fr = null;
		
		// possible writers
		ObjectOutputStream oos = null;
		FrameWriter fw = null;
		
		switch (inFormat) {
		case SAMPLE: ois = new ObjectInputStream(System.in); break;
		case FRAME: fr = new FrameReader(); fd = fr.getFrameSize(); break;
		case ASCII:
		case ASCII_L:
			br = new BufferedReader(new InputStreamReader(System.in));
		}
		
		
		if (outFormat == Format.SAMPLE)
			oos = new ObjectOutputStream(System.out);
		
		double [] buf = new double [fd];
		byte [] label = null;
		
		// read until done...
		while (true) {
			
			Sample s = null;
			
			// try to read...
			switch (inFormat) {
			case SAMPLE: 
				s = (Sample) ois.readObject();
				break;
			case FRAME:
				if (fr.read(buf))
					s = new Sample(0, buf);
				break;
			case LFV:
				label = new byte [LABEL_SIZE];
			case UFV:
				if (!read(System.in, buf, label)) 
					break;
				
				int length = 0;
				while (label[length] != 0)
					length++;
				String ls = (label == null ? null : new String(label, 0, length, "ASCII"));
				s = new Sample(label == null ? 0 : lookup1.get(ls), buf);

				break;
			case ASCII:
			case ASCII_L:
				String line = br.readLine();
				if (line == null)
					break;
				String [] cols = line.trim().split("\\s+");
				int i1 = 0;
				ls = null;
				
				if (inFormat == Format.ASCII_L)
					ls = cols[i1++];
				
				if (!lookup1.containsKey(ls)) {
					lookup1.put(ls, lab);
					lookup2.put(lab, ls);
					lab++;
				}
				
				int i2 = 0;
				buf = new double [cols.length - i1];
				for (; i1 < cols.length; ++i1)
					buf[i2++] = Double.parseDouble(cols[i1]);
				
				s = new Sample(ls == null ? 0 : lookup1.get(ls), buf);
			}
			
			// anything read?
			if (s == null)
				break;
			
			// write out...
			switch (outFormat) {
			case SAMPLE:
				oos.writeObject(s);
				break;
			case FRAME:
				if (fw == null)
					fw = new FrameWriter(buf.length);
				fw.write(buf);
				break;
			case LFV:
				byte [] outlabel1 = new byte [LABEL_SIZE];
				byte [] outlabel2 = (s.c == 0 ? "UNSET".getBytes("ASCII") : lookup2.get(s.c).getBytes("ASCII"));
				for (int i = 0; i < LABEL_SIZE; ++i) {
					if (i < outlabel2.length)
						outlabel1[i] = outlabel2[i];
					else
						outlabel1[i] = 0;
				}
				System.out.write(outlabel1);
			case UFV:
				ByteBuffer bb = ByteBuffer.allocate(buf.length * Float.SIZE/8);
				
				// UFVs are little endian!
				bb.order(ByteOrder.LITTLE_ENDIAN);
				
				for (double d : buf) 
					bb.putFloat((float) d);
				
				System.out.write(bb.array());
				break;
			case ASCII_L:
				String lab = lookup2.get(s.c);
				System.out.print((lab == null ? "UNSET" : lab) + "\t");
			case ASCII:
				for (int i = 0; i < buf.length; ++i) {
					System.out.print(buf[i]);
					if (i < buf.length - 1)
						System.out.print("\t");
					else
						System.out.println();
				}
				break;
			}
		}
		
		// be nice, close everything
		if (fw != null)
			fw.close();
		
		if (oos != null)
			oos.close();
	}
	
	public static boolean read(InputStream is, double [] buf, byte [] label) throws IOException {
		byte [] rb = new byte [buf.length * Float.SIZE/8];
		
		int read = 0;;
		
		if (label != null)
			read = is.read(label);
		
		read = is.read(rb);
		
		// complete frame?
		if (read <  buf.length)
			return false;
		
		// decode the double
		ByteBuffer bb = ByteBuffer.wrap(rb);
		bb.order(ByteOrder.LITTLE_ENDIAN);
		
		for (int i = 0; i < buf.length; ++i)
			buf[i] = (double) bb.getFloat();
		
		return true;
	}
}
