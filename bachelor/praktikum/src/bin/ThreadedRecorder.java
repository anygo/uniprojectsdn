package bin;

import sampled.AudioCapture;
import java.io.*;


public class ThreadedRecorder implements Runnable {
	/** AudioCapture to read from, needs to be fully initialized */
	private AudioCapture source;
	
	/** is the recorder paused? */
	private volatile boolean paused = false;
	
	/** is a stop requested? */
	private volatile boolean stopRequested = false;
	
	/** is the capture finished? indicator to the recording thread */
	private volatile boolean finished = false;
	
	/** output stream to write to */
	private OutputStream os = null;
	
	/**
	 * Create a new ThreadedRecorder using an initialized AudioCapture object.
	 * Once start is called, the raw data (matching the AudioCapture format) will
	 * be saved to a file.
	 * @param source
	 */
	public ThreadedRecorder(AudioCapture source) {
		this.source = source;
	}
	
	/**
	 * Start the recording and save to the given file. File will be overwritten!
	 * @param fileName absolute path to target file; if null, a ByteArrayOutputStream is used.
	 * @return the output stream used for this recording; either BufferedOuputStream or ByteArrayOutputStream
	 */
	public OutputStream start(String fileName) throws IOException {
		// make sure we don't cancel a recording
		if (os != null)
			throw new IOException("Recorder already running!");
		
		if (fileName == null)
			os = new ByteArrayOutputStream();
		else
			os = new BufferedOutputStream(new FileOutputStream(fileName));

		// yeah, start off
		new Thread(this).start();
		
		return os;
	}
	
	/** 
	 * (un)pause the recording; on pause, the recording continues, but is not
	 * saved to the file.
	 */
	public void pause() {
		paused = !paused;
	}
	
	/***
	 * Stop the recording
	 */
	public void stop() {
		stopRequested = true;
	}
	
	/** 
	 * Internal thread function, takes care of the actual recording
	 */
	public void run() {
		// enable internal buffer
		source.enableInternalBuffer(0);
		finished = false;
		
		try {
			// run as long as we want...
			while (!stopRequested) {
				source.read(null); // read to internal buffer
				if (!paused) {
					os.write(source.getRawBuffer());
				}
			}

			// close and cleanup
			os.close();
		} catch (IOException e) {
			System.err.println("ThreadedRecorder.run(): I/O error: " + e.toString());
		} catch (Exception e) {
			System.err.println("ThreadedRecorder.run(): " + e.toString());
		} finally {
			// note to main thread
			finished = true;
			stopRequested = false;
			paused = false;
			os = null;
		}
	}
	
	public void tearDown() throws IOException {
		source.tearDown();
	}
	
	public boolean isRecording() {
		return !finished;
	}
	
	public boolean isPaused() {
		return paused;
	}
	
	public static void main(String[] args) throws IOException {
		if (args.length != 1) {
			System.err.println("usage: bin.ThreadedRecorder <mixer-name>");
			System.exit(1);
		}
		
		ThreadedRecorder tr = new ThreadedRecorder(new AudioCapture(args[0], 16, 16000));
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		
		String h = "Commands:\n" +
				"  's <filename>' : start recording\n" +
				"  'p' : pause recording\n" +
				"  'e' : finish recording\n" +
				"  'h' : display this help";
		
		System.out.println(h);

		String line;
		while (true) {
			line = br.readLine();
			
			if (line == null)
				break;
			else if (line.toUpperCase().startsWith("E")) {
				System.out.println("Recording finished...");
				tr.stop();
			} else if (line.toUpperCase().startsWith("P")) {
				System.out.println("Recording (un)paused...");
				tr.pause();
			} else if (line.toUpperCase().startsWith("S ")) {
				System.out.println(" " + line.substring(2));
				tr.start(line.substring(2));
			} else if (line.toUpperCase().startsWith("H"))
				System.out.println(h);
		}
	}
}
