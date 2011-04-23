package visual;

import sampled.TinyAudioSource;

public interface Buffer {
	
	public double[] read(int startIndex, int howMany);
	
	public int read(double[] buf, int startIndex);
	
	public int getBufferSize();
	
	public TinyAudioSource createTinyAudioSource(int startindex, int howMany);

}