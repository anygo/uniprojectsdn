package visual;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.swing.JComponent;

import framed.*;

public final class Visualizer2D extends JComponent implements FrameSource {
	private static final long serialVersionUID = 1L;
	
	public static Dimension DEFAULT_SIZE = new Dimension(512, 256);
	
	private FrameSource source = null;
	
	private int fs = 0;
	private int bs = 0;
	
	private BufferedImage blt = null;
	
	private String title = null;
	
	private boolean log = false;
	
	public Visualizer2D(FrameSource source, String title, boolean log) {
		this.source = source;
		this.log = log;
		bs = DEFAULT_SIZE.width;
		fs = source.getFrameSize();
		this.title = title;
		
		// initialize the ring buffer
		ringbuf = new double [bs][fs];
		setPreferredSize(DEFAULT_SIZE);
		
		// initialize the image for 
		blt = new BufferedImage(DEFAULT_SIZE.width, DEFAULT_SIZE.height, BufferedImage.TYPE_INT_ARGB);
	}
	
	public boolean getLog() {
		return log;
	}
	
	public void setLog(boolean log) {
		this.log = log;
	}
	

	public int getFrameSize() {
		return fs;
	}

	private double min = Double.MAX_VALUE;
	private double max = Double.MIN_VALUE;
	
	public void bufferedPaint(BufferedImage img) {
		for (int i = 0; i < bs; ++i) {
			// start at oldest value
			int c = (ind + i) % bs;
			
			for (int j = 0; j < img.getHeight(); ++j) {
				// no interpolation, just indexing
				int ip = (int) (j * fs / (double) img.getHeight());
				double v = (ringbuf[c][ip] - min) / (max - min);
				
				if (v > 1.)
					v = 1.;
				
				int gray = (int)((1.-v) * 255);
				int rgb = 0xFF << 24 
					| gray << 16
					| gray << 8
					| gray;
				img.setRGB(i, img.getHeight() - j - 1, rgb);
			
			}
		}
	}
	
	/** min/max value BEFORE taking logarithm */
	public void setMinMax(double min, double max) {
		this.max = (log ? Math.log(1E-6 + max) : max);
		this.min = (log ? Math.log(1E-6 + min) : min);
	}
	
	public double getMax() { return max; }
	public double getMin() { return min; }
	
	public void paint(Graphics g) {
		bufferedPaint(blt);
		g.drawImage(blt, 0, 0, this);
		g.setColor(Color.RED);
		g.drawString(title, 5, 10);
	}
	
	private int ind = 0;
	
	private double [][] ringbuf = null;
	
	public boolean read(double[] buf) throws IOException {
		// read from source
		if (!source.read(ringbuf[ind]))
			return false;
		
		// save to outgoing buffer
		System.arraycopy(ringbuf[ind], 0, buf, 0, buf.length);
		
		if (log)
			for (int i = 0; i < fs; ++i)
				ringbuf[ind][i] = Math.log(1E-6 + ringbuf[ind][i]);
		
		// repaint the scene
		repaint();
			
		// increment index
		ind = (ind + 1) % bs;
		
		return true;
	}

}
