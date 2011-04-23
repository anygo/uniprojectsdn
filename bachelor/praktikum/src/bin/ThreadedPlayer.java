package bin;

import java.io.IOException;

import sampled.AudioFileReader;
import sampled.AudioPlay;
import sampled.RawAudioFormat;

public class ThreadedPlayer implements Runnable {

	private AudioPlay player;

	/** is the player paused? */
	private volatile boolean paused = false;

	/** is a stop requested? */
	private volatile boolean stopRequested = false;

	/** is the playback finished? indicator to the player thread */
	private volatile boolean finished = false;

	/**
	 * Create a threaded player using the given audio player.
	 * @param player
	 */
	public ThreadedPlayer(AudioPlay player) {
		this.player = player;		
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
	
	public void start() throws IOException {
		new Thread(this).start();
	}

	public void run() {
		try {
			while (!stopRequested) {
				if (paused)
					continue;
				if (player.write() <= 0)
					break;
			}
			
			player.tearDown();
		} catch (IOException e) {
			System.err.println("ThreadedPlayer.run(): I/O error: " + e.toString());
		} catch (Exception e) {
			System.err.println("ThreadedPlayer.run(): " + e.toString());
		} finally {
			// note to main thread
			finished = true;
			stopRequested = false;
			paused = false;
		}

	}
	
	public boolean isPlaying() {
		return !finished;
	}
	
	public boolean isPaused() {
		return paused;
	}

	public static void main(String [] args) throws Exception {
		for (String file : args) {
			System.err.println("Hit [ENTER] to start playing " + file);
			System.in.read();
			
			ThreadedPlayer play = new ThreadedPlayer(new AudioPlay(new AudioFileReader(file, RawAudioFormat.getAudioFormat("ssg/16"), true)));
			play.start();
		}
	}
}