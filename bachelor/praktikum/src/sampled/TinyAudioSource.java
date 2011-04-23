package sampled;

import java.io.IOException;
import visual.BufferedAudioSource;

public class TinyAudioSource implements AudioSource {
	double buf[];
	BufferedAudioSource bas;
	int alreadyRead = 0;
	
	public TinyAudioSource(double[] buf, BufferedAudioSource bas) {
		this.buf = new double[buf.length];
		System.arraycopy(buf, 0, this.buf, 0, buf.length);
		this.bas = bas;
	}

	@Override
	public boolean getPreEmphasis() {
		return bas.getPreEmphasis();
	}

	@Override
	public int getSampleRate() {
		return bas.getSampleRate();
	}

	@Override
	public int read(double[] buf) throws IOException {
		if (alreadyRead >= this.buf.length) {
			return 0;
		}
		int i = 0;
		for(; i < buf.length && i + alreadyRead < this.buf.length ; i++, alreadyRead++) {
			buf[i] = this.buf[alreadyRead+i];
		}
			
		//System.arraycopy(this.buf, alreadyRead, buf, 0, buf.length);
		return i;
	}

	@Override
	public void setPreEmphasis(boolean applyPreEmphasis, double a) {
		bas.setPreEmphasis(applyPreEmphasis, a);
		
	}

	@Override
	public void tearDown() throws IOException {
		bas.tearDown();
		
	}
}