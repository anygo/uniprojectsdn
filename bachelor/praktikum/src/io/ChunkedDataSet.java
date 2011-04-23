package io;

import java.util.*;
import java.io.*;

import statistics.Sample;

/**
 * Use the ChunkedDataSet if you have a list of files containing framed data and
 * want to read that in sequencially.
 * @author sikoried
 *
 */
public class ChunkedDataSet {
	private LinkedList<String> validFiles = new LinkedList<String> ();
	private int cind = 0;
	
	/**
	 * A chunk consists of its name and a (ready-to-read) FrameReader
	 * @author sikoried
	 *
	 */
	public static class Chunk {
		/**
		 * Create a new Chunk and prepare the FrameReader to read from the given
		 * file.
		 * 
		 * @param fileName
		 * @throws IOException
		 */
		public Chunk(String fileName) throws IOException {
			// prepare the reader
			reader = new FrameReader(fileName);
			
			// get the filename
			name = fileName.substring(fileName.lastIndexOf(System.getProperty("file.separator")) + 1);
		}
		
		/** Ready-to-use FrameReader */
		public FrameReader reader;
		
		/** file name without the path */
		public String name;
	}
	
	/**
	 * Get the next Chunk from the list.
	 * @return Chunk instance on success, null if there's no more chunks
	 * @throws IOException
	 */
	public synchronized Chunk nextChunk() throws IOException {
		if (cind < validFiles.size())
			return new Chunk(validFiles.get(cind++));
		return null;
	}
	
	public synchronized void rewind() {
		cind = 0;
	}
	
	/** 
	 * Create a ChunkDataSet using the given list file.
	 * @param fileName path to the list file
	 * @throws IOException
	 */
	public ChunkedDataSet(String fileName) 
		throws IOException {
		setChunkList(fileName);
	}
	
	public ChunkedDataSet(List<String> fileNames) {
		validFiles.addAll(fileNames);
	}
	
	/**
	 * Get the number of Chunks in this data set
	 * @return
	 */
	public int numberOfChunks() {
		return validFiles.size();
	}
	
	/**
	 * Load the given chunk list.
	 * @param fileName
	 * @throws IOException
	 */
	public void setChunkList(String fileName) throws IOException {
		validFiles.clear();
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String name;
		while ((name = br.readLine()) != null) {
			File test = new File(name);
			if (test.canRead())
				validFiles.add(name);
		}
	}
	
	/**
	 * Cache all chunks into a List<Sample> for easier (single-core) access
	 * @return
	 * @throws IOException
	 */
	public synchronized List<Sample> cachedData() throws IOException {
		// remember old index
		int oldInd = cind;
		cind = 0;
		
		LinkedList<Sample> data = new LinkedList<Sample>();
		Chunk chunk;
		while ((chunk = nextChunk()) != null) {
			double [] buf = new double [chunk.reader.getFrameSize()];
			while (chunk.reader.read(buf))
				data.add(new Sample(0, buf));
		}
		
		// restore old index
		cind = oldInd;
		
		return data;
	}
}
