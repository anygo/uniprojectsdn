package util;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class IOUtil {

	/**
	 * Reads floats from Binary file
	 * 
	 * @param file
	 *            : Name of File to read from
	 * @param buf
	 *            : Write into this buffer
	 * @return true on success, false else.
	 * @throws IOException
	 */
	public static boolean readFloatsFromBinaryFile(String file, double[] buf,
			ByteOrder bo) throws IOException {
		byte[] rb = new byte[buf.length * Float.SIZE / 8];
		BufferedInputStream bis = new BufferedInputStream(new FileInputStream(
				file));
		int read = bis.read(rb);

		// complete frame?
		if (read < buf.length)
			return false;

		// decode the double
		ByteBuffer bb = ByteBuffer.wrap(rb);
		bb.order(bo);

		for (int i = 0; i < buf.length; ++i) {
			buf[i] = (double) bb.getFloat();
		}
		
		bis.close();
		return true;
	}

	/**
	 * Reads floats from ASCII file
	 * 
	 * @param file
	 *            : Name of File to read from
	 * @param buf
	 *            : Write into this buffer
	 * @return true on success, false else.
	 * @throws IOException
	 */
	public static boolean readFloatsFromAsciiFile(String file, double[] buf)
			throws IOException {

		FileReader fr = new FileReader(file);
		StringBuffer sb = new StringBuffer();
		int c;
		while (((c = fr.read()) != -1)) {
			sb.append((char) c);
		}

		String[] tokens = sb.toString().trim().split("\\s+"); // Separated by
																// "whitespace"
		if (tokens.length != buf.length) {
			throw new IOException(
					"Number of floats in file do not match given buffer size! buf.length: "
							+ buf.length + " tokens.length: " + tokens.length);
		}
		int i = 0;
		for (String t : tokens) {
			buf[i] = Double.valueOf(t);
			i++;
		}

		fr.close();
		return true;
	}
}
