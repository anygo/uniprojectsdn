package io;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class FrameWriter {
	
	/** OutputStream to write to */
	private OutputStream os = System.out;
	
	/** frame size to write */
	private int fs = 0;
	
	/** write ufv? 4-byte floats! */
	private boolean ufv = false;

	/**
	 * Generate a FrameWriter that writes to stdout
	 * @param frameSize size of output frames
	 * @throws IOException
	 */
	public FrameWriter(int frameSize) throws IOException {
		fs = frameSize;
		initialize();
	}
	
	/**
	 * Generate a FrameWriter that writes to stdout
	 * @param frameSize size of output frames
	 * @throws IOException
	 */
	public FrameWriter(int frameSize, boolean ufv) throws IOException {
		fs = frameSize;
		this.ufv = true;
		initialize();
	}
	
	/**
	 * Generate a FrameWriter that writes to the given file
	 * @param frameSize
	 * @param fileName
	 * @throws IOException
	 */
	public FrameWriter(int frameSize, String fileName) throws IOException {
		fs = frameSize;
		if (fileName != null)
			os = new BufferedOutputStream(new FileOutputStream(fileName));
		initialize();
	}
	
	/**
	 * Generate a FrameWriter that writes to the given file
	 * @param frameSize
	 * @param fileName
	 * @throws IOException
	 */
	public FrameWriter(int frameSize, String fileName, boolean ufv) throws IOException {
		fs = frameSize;
		this.ufv = ufv;
		os = new BufferedOutputStream(new FileOutputStream(fileName));
		initialize();
	}
	
	/**
	 * Initialize the output stream by writing out the frame size
	 * @throws IOException
	 */
	private void initialize() throws IOException {
		// no initialization required for UFV
		if (ufv)
			return;
		
		// write out the frame length
		ByteBuffer bb = ByteBuffer.allocate(Integer.SIZE/8);
		bb.order(ByteOrder.LITTLE_ENDIAN);
		bb.putInt(fs);
		os.write(bb.array());
	}
	
	/**
	 * To write the frame, pack the doubles in a byte array
	 * @param buf
	 * @throws IOException
	 */
	public void write(double [] buf) throws IOException {
		if (ufv)
			writeUFV(os, buf);
		else
			writeDoubleArray(os, buf);
	}
	
	public void close() throws IOException {
		if (!(os == System.out || os == System.err))
			os.close();
	}
	
	/**
	 * In the end, close the data file to prevent data loss!
	 */
	public void finalize() throws Throwable{
		try { 
			os.close();
		} finally {
			super.finalize();
		}
	}
	
	/**
	 * Write a double array to the given stream (raw)
	 * @param os OutputStream to use
	 * @param buf 
	 * @throws IOException
	 */
	public static void writeDoubleArray(OutputStream os, double [] buf) 
		throws IOException {
		ByteBuffer bb = ByteBuffer.allocate(buf.length * Double.SIZE/8);
		bb.order(ByteOrder.LITTLE_ENDIAN);
		
		for (double d : buf) 
			bb.putDouble(d);
		
		os.write(bb.array());
	}
	
	/**
	 * Write UFVs: blocks of floats, for compatibility with the old recognition
	 * system
	 * 
	 * @param os
	 * @param buf
	 * @throws IOException
	 */
	public static void writeUFV(OutputStream os, double [] buf)
		throws IOException {
			ByteBuffer bb = ByteBuffer.allocate(buf.length * Float.SIZE/8);
			
			// UFVs are little endian!
			bb.order(ByteOrder.LITTLE_ENDIAN);
			
			for (double d : buf) 
				bb.putFloat((float) d);
			
			os.write(bb.array());
	}
}
