package sampled;

import java.util.*;
import java.io.*;

public class AudioFileListReader implements AudioSource {

	private AudioFileReader current = null;
	
	private RawAudioFormat format = null;
	
	private String fileList = null;
	
	/** list of files, FIFO */
	private Queue<String> list = new LinkedList<String>();
	
	/** use buffered reader? */
	private boolean cache = true;
	
	/** apply pre-emphasis? */
	private boolean preemphasize = true;
	
	/** pre-emphasis factor */
	private double a = AudioFileReader.DEFAULT_PREEMPHASIS_FACTOR;
	
	/** 
	 * Generate a new AudioFileListReader using given file list, RawAudioFormat
	 * and cache indicator. 
	 * @param fileList
	 * @param format
	 * @param cache
	 * @throws IOException
	 * 
	 * @see RawAudioFormat.create
	 */
	public AudioFileListReader(String fileList, RawAudioFormat format, boolean cache) throws IOException {
		this.fileList = fileList;
		this.format = format;
		this.cache = cache;
		initialize();
	}
	
	/**
	 * Read the given list file line-by-line, check if they exist, and add them
	 * to the file list
	 * @throws IOException
	 */
	private void initialize() throws IOException {
		
		// read the file list
		BufferedReader br = new BufferedReader(new FileReader(fileList));
		String buf = null;
		while ((buf = br.readLine()) != null) {
			try {
				File f = new File(buf);
				
				// check file
				if (!f.exists())
					throw new FileNotFoundException();
				if (!f.canRead())
					throw new IOException();
				
				// add to list
				list.add(buf);
			} catch (FileNotFoundException e) {
				System.err.println("skipping file '" + buf + "': file not found");
			} catch (IOException e) {
				System.err.println("skipping file '" + buf + "': permission denied");
			} 			
		}
		
		if (list.size() == 0)
			throw new IOException("file list was empty!");
		
		// load the first file
		current = new AudioFileReader(list.poll(), format, cache);
		current.setPreEmphasis(preemphasize, a);
	}
	
	public boolean getPreEmphasis() {
		if (current != null)
			return getPreEmphasis();
		else
			return preemphasize;
	}
	
	public void setPreEmphasis(boolean applyPreEmphasis, double a) {
		preemphasize = applyPreEmphasis;
		this.a = a;
		if (current != null)
			current.setPreEmphasis(applyPreEmphasis, a);
	}
	
	public String toString() {
		return "AudioFileListReader: " + fileList + " (" + list.size() + " valid files\n" + format.toString();
	}
	
	public int read(double[] buf) throws IOException {
		if (current == null)
			return 0;
		
		int read = current.read(buf);
		
		
		if (read == buf.length) {
			// enough samples read
			return read;
		}
		else if (read == 0) {
			// no samples read, load next file if possible
			if (list.size() == 0)
				return 0;
			
			// load next file
			current.tearDown();
			current = new AudioFileReader(list.remove(), format, cache);
			current.setPreEmphasis(preemphasize, a);
			
			// read frame
			return current.read(buf);
		} else {
			// early EOF, pad with zeros, load next file
			for (int i = read; i < buf.length; ++i)
				buf[i] = 0.;
			
			if (list.size() > 0) {
				current = new AudioFileReader(list.remove(), format, cache);
				current.setPreEmphasis(preemphasize, a);
			}
			
			return buf.length;
		}
	}
	
	public void tearDown() throws IOException {
		// nothing to do here
	}

	public int getSampleRate() {
		if (current != null)
			return current.getSampleRate();
		else
			return 0;
	}

}
