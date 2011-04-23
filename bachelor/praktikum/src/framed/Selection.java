package framed;

import io.FrameReader;
import io.FrameWriter;

import java.io.IOException;
import java.util.ArrayList;
import exceptions.*;

public class Selection implements FrameSource {

	/** FrameSource to read from */
	private FrameSource source = null;
	
	/** indices of the feature dimensions to select; default: MFCC0-11 */
	private int [] indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
	
	/** internal read buffer */
	private double [] buf = null;
	
	/** incoming frame size */
	private int fs_in = 0;
	
	/** outbound frame size */
	private int fs_out = indices.length;
	
	/**
	 * Generate a default feature selection: dimensions 0-11 (standard mfcc)
	 * @param source
	 */
	public Selection(FrameSource source) {
		this.source = source;
		fs_in = source.getFrameSize();
		buf = new double [fs_in];
	}
	
	/**
	 * Select the first n coefficients
	 * @param source
	 */
	public Selection(FrameSource source, int n) {
		this.source = source;
		indices = new int [n];
		for (int i = 0; i < n; ++i)
			indices[i] = i;
		fs_out = n;
		fs_in = source.getFrameSize();
		buf = new double [fs_in];
	}
	
	/**
	 * Apply a selection to the incoming feature frame
	 * @param source
	 * @param indices a implicit mapping, place the indices of the desired dimensions here
	 */
	public Selection(FrameSource source, int [] indices) {
		this(source);
		this.indices = indices;
		fs_out = indices.length;
	}
	
	/** 
	 * Return the outgoing frame size 
	 */
	public int getFrameSize() {
		return fs_out;
	}
	
	public String toString() {
		StringBuffer buf = new StringBuffer();
		buf.append("selection fs_in=" + fs_in + " fs_out=" + fs_out + " mapping_in_out=[");
		for (int i = 0; i < fs_out; ++i)
			buf.append(" " + indices[i] + "->" + i);
		return buf.toString() + " ]";
	}

	/**
	 * Read the next frame and transfer the features to the outgoing buffer
	 * according to the indices
	 */
	public boolean read(double[] buf) throws IOException {
		if (!source.read(this.buf))
			return false;
		
		// copy; go the long way, there might be re-ordering!
		for (int i = 0; i < buf.length; ++i)
			buf[i] = this.buf[indices[i]];
		
		return true;
	}
	
	/** 
	 * Create a Selection object according to the parameter string and
	 * attach it to the source.
	 * @param source framesource to read from
	 * @param formatString comma separated list of indices or ranges (e.g. "0,1,4-8")
	 * @return ready-to-use Selection
	 */
	public static Selection create(FrameSource source, String formatString)
		throws MalformedParameterStringException {
		ArrayList<Integer> indices = new ArrayList<Integer>();
		String [] parts = formatString.split(",");
		
		try {
			for (String i : parts) {
				if (i.indexOf("-") > 0) {
					String [] range = i.split("-");
					int start = Integer.parseInt(range[0]);
					int end = Integer.parseInt(range[1]);
					for (int j = start; j <= end; ++j)
						indices.add(j);
				} else {
					indices.add(Integer.parseInt(i));
				}
			}
		} catch (Exception e) {
			// something went wrong analyzing the parameter string
			throw new MalformedParameterStringException(e.toString());
		}
		
		if (indices.size() < 1)
			throw new MalformedParameterStringException("No indices in format string!");
		
		// generate the final indices array
		int [] ind = new int [indices.size()];
		for (int i = 0; i < ind.length; ++i)
			ind[i] = indices.get(i);
		
		// generate the Selection object
		return new Selection(source, ind);
	}
	
	public static String synopsis = 
		"usage: framed.Selection [--ufv-in dim] [--ufv-out | --ascii-out] -s <ParameterString> < infile > outfile\n" +
		"ParameterString: comma separated list of individual digits (0,1,...) or ranges (4-12), e.g. \"0,1,4-8\"\n";
	
	/**
	 * main program; for usage see synopsis
	 * @param args
	 * @throws IOException
	 * @throws IOException, MalformedParameterStringException
	 */
	public static void main(String [] args) throws IOException, MalformedParameterStringException {
		if (args.length < 1) {
			System.err.println(synopsis);
			System.exit(1);
		}
		
		boolean ascii = false;
		boolean ufvout = false;
		boolean ufvin = false;
		
		int ufvdim = 0;
		
		String parameter = null;
		
		for (int i = 0; i < args.length; ++i) {
			if (args[i].equals("--ascii-out"))
				ascii = true;
			else if (args[i].equals("--ufv-out"))
				ufvout = true;
			else if (args[i].equals("--ufv-in")) {
				ufvin = true;
				ufvdim = Integer.parseInt(args[++i]);
			} else if (args[i].equals("-s"))
				parameter = args[++i];
		}
		
		if ((ascii && ufvout) || parameter == null) {
			System.err.println(synopsis);
			System.exit(1);
		}
		
		// get a STDIN frame reader
		FrameReader fr = (ufvin ? new FrameReader(null, true, ufvdim) : new FrameReader());
		
		// attach to selection
		FrameSource selection = Selection.create(fr, parameter);
		
		double [] buf = new double [selection.getFrameSize()];
		
		// get a STDOUT frame writer if required
		FrameWriter fw = null;
		if (!ascii) {
			fw = (ufvout ? new FrameWriter(buf.length, true) : new FrameWriter(buf.length));
		}
		
		// read the frames, select
		while (selection.read(buf)) {
			if (ascii) {
				// ascii?
				for (int i = 0; i < buf.length - 1; ++i)
					System.out.print(buf[i] + " ");
				System.out.println(buf[buf.length-1]);
			} else {
				// binary, writer takes care of UFV
				fw.write(buf);
			}
		}
	}
}
