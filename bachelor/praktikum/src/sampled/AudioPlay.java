package sampled;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.Mixer;
import javax.sound.sampled.SourceDataLine;


public class AudioPlay {
	private SourceDataLine line;
	
	private int bs;
	private double [] buf = null;
	
	private String mixerName = null;
	private AudioSource source = null;
	
	/// output bit rate
	public static final int BITRATE = 16;
	
	/// buffer length in msec
	public static final int BUFLENGTH = 10;

	private double scale = 1.;
	
	private ByteArrayOutputStream baos = new ByteArrayOutputStream();
	private DataOutputStream dos = new DataOutputStream(baos);
	
	public AudioPlay(AudioSource source) 
		throws IOException, LineUnavailableException {
		this(null, source);
	}
	
	/**
	 * Creates a new AudioPlay object and initializes it.
	 * 
	 * @param mixerName name of mixer to use for play back (or null for default)
	 * @param format audio format of the 
	 * @param buffer Buffer to read from
	 * @throws IOException
	 * @throws LineUnavailableException
	 */
	public AudioPlay(String mixerName, AudioSource source)
		throws IOException, LineUnavailableException {
		
		this.mixerName = mixerName;
		this.source = source;

		initialize();
	}

	/** 
	 * Initialize the play back by setting up the outgoing lines.
	 * @throws IOException
	 * @throws LineUnavailableException
	 */
	private void initialize() throws IOException, LineUnavailableException {
		// standard linear PCM at 16 bit and the availabpe sample rate
		AudioFormat af = new AudioFormat(source.getSampleRate(), BITRATE, 1, true, true);
		DataLine.Info info = new DataLine.Info(SourceDataLine.class, af);
		
		// No mixer specified, use default mixer
		if (mixerName == null) {
			line = (SourceDataLine) AudioSystem.getLine(info);
		} else {
			// mixerName specified, use this Mixer to write to 
			Mixer.Info [] availableMixers = AudioSystem.getMixerInfo();
			Mixer.Info target = null;
			for (Mixer.Info m : availableMixers)
				if (m.getName().trim().equals(mixerName))
					target = m;
			
			// If no target, fall back to default line
			if (target != null)
				line = (SourceDataLine) AudioSystem.getMixer(target).getLine(info);
		}
		
		line.open(af);
		line.start();
		
		// init the buffer		
		bs = (int) (BUFLENGTH * af.getSampleRate() / 1000);
		buf = new double [bs];
		
		scale = Math.pow(2, BITRATE - 1);
	}
	
	/**
	 * 
	 * @return String of mixerName
	 */
	public String getMixerName(){
		if(mixerName != null)
			return mixerName;
		else
			return "default mixer";
	}

	/**
	 * Close everyting
	 * @throws IOException
	 */
	public void tearDown() throws IOException {
		line.drain();
		line.stop();
		line.close();
		source.tearDown();
	}

	/**
	 * write one frame from data array to audioSource (playback)
	 * @return number of bytes played(written to audioSource) or -1 if audiobuffer is empty
	 * @throws IOException
	 */
	public int write() throws IOException {
		int count = source.read(buf);
		
		if (count <= 0) {
			tearDown();
			return 0;
		}
		
		// set rest to zero
		if (count < bs)
			for (int i = count; i < bs; ++i)
				buf[i] = 0;
		
		// double -> short conversion
		for (int i = 0; i < bs; ++i)
			dos.writeShort((short)(buf[i] * scale));
		dos.flush();
		
		byte [] outgoing = baos.toByteArray();  
		count = line.write(outgoing, 0, outgoing.length);
		
		// reset the conversion stream
		baos.reset();
		
		return count;
	}

	protected void finalize() throws Throwable {
		try {
			tearDown();
		} finally {
			super.finalize();
		}
	}
	
	public static void main(String [] args) throws Exception {
		for (String file : args) {
			System.err.println("Now playing " + file);
			AudioPlay play = new AudioPlay("default [plughw:1,0]", new AudioFileReader(file, RawAudioFormat.getAudioFormat("ssg/16"), true));
			while (play.write() > 0)
				;
		}
	}
}