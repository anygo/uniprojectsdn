package io;

import java.io.*;
import java.nio.*;

/**
 * Read frames from a file or stdin (unlabeled feature data), first int (4byte) 
 * is the frame size.
 * 
 * @author sikoried
 *
 */
public class FrameReader implements framed.FrameSource {

	/** incoming frame size */
	private int fs = 0;
	
	/** input stream to read from */
	private InputStream is = System.in;
	
	/** file name, if any */
	private String fileName = null;
	
	/** read ufvs? */
	private boolean ufv = false;
	
	public FrameReader() throws IOException {
		initialize();
	}
	
	public FrameReader(String fileName) throws IOException {
		if (fileName != null)
			is = new BufferedInputStream(new FileInputStream(fileName));
		this.fileName = fileName;
		initialize();
	}
	
	public FrameReader(String fileName, boolean ufv, int fs) throws IOException {
		if (fileName != null)
			is = new BufferedInputStream(new FileInputStream(fileName));
		this.ufv = ufv;
		this.fs = fs;
		this.fileName = fileName;
		initialize();
	}
	
	/**
	 * Read an integer to determine the frame size
	 * @throws IOException
	 */
	private void initialize() throws IOException {
		if (ufv)
			return;
		
		byte [] fs_raw = new byte [Integer.SIZE/8];
		// try to read the first int
		if (is.read(fs_raw) != Integer.SIZE/8)
			throw new IOException("couldn't initialize FrameReader");
		
		// determine the frame size
		ByteBuffer bb = ByteBuffer.wrap(fs_raw);
		bb.order(ByteOrder.LITTLE_ENDIAN);
		
		fs = bb.getInt();
	}
	
	/**
	 * Close the FrameReader's input file
	 * @throws IOException
	 */
	public void close() throws IOException {
		is.close();
	}
	
	/**
	 * Return the size of the output frames
	 */
	public int getFrameSize() {
		return fs;
	}
	
	public String toString() {
		return "FrameReader: source=" + (fileName == null ? "stdin" : fileName) + " frame_size=" + fs;
	}
	
	/**
	 * Read the next frame, convert the raw data to doubles
	 */
	public boolean read(double[] buf) throws IOException {
		boolean status;
		if (ufv)
			status = readUFVArray(is, buf);
		else
			status = readDoubleArray(is, buf);
		
		// no more frames, close file!
		if (status == false)
			is.close();
		
		return status;
	}
	
	/**
	 * Read a double array from the given input stream (expects raw doubles...)
	 * @param is InputStream to use
	 * @param buf Buffer to save values to
	 * @return true on success
	 * @throws IOException
	 */
	public static boolean readDoubleArray(InputStream is, double [] buf) throws IOException {
		byte [] rb = new byte [buf.length * Double.SIZE/8];
		int read = is.read(rb);
		
		// complete frame?
		if (read <  buf.length)
			return false;
		
		// decode the double
		ByteBuffer bb = ByteBuffer.wrap(rb);
		bb.order(ByteOrder.LITTLE_ENDIAN);
		
		for (int i = 0; i < buf.length; ++i)
			buf[i] = bb.getDouble();
		
		return true;
	}
	
	public static boolean readUFVArray(InputStream is, double [] buf) throws IOException {
		byte [] rb = new byte [buf.length * Float.SIZE/8];
		int read = is.read(rb);
		
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
	
	/// make sure the input file is closed!
	protected void finalize() throws Throwable {
		try {
			is.close();
		} finally {
			super.finalize();
		}
	}
	
	public static void main(String [] args) throws IOException {
		if (args.length < 1) {
			System.err.println("usage: framed.FrameReader <file1> [file2 ...]");
			System.exit(1);
		}
		
		for (String f : args) {
			FrameReader fr = new FrameReader(f);
			double [] buf = new double [fr.getFrameSize()];
			while (fr.read(buf)) {
				for (int i = 0; i < buf.length-1; ++i) 
					System.out.print(buf[i] + " ");
				System.out.println(buf[buf.length-1]);
			}
		}
	}
}
