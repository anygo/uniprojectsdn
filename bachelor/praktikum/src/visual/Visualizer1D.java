package visual;

import framed.*;

import java.io.IOException;
import javax.swing.JComponent;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Image;

public class Visualizer1D extends JComponent implements FrameSource {
	private static final long serialVersionUID = 1L;
	
	/** Frame source to read from */
	protected FrameSource source = null;
	
	/** incoming frame size */
	protected int fs = 0;
	
	/** internal buffer */
	protected double [] buf = null;
	
	/** title of the plot */
	protected String title = null;
	
	/** apply log to the data? */
	protected boolean log = false;
	
	public static Dimension DEFAULT_SIZE = new Dimension(512, 256);
	
	/**
	 * Create a new FrameVisualizer
	 * @param source FrameSource to read from
	 * @param title Title of the JFrame
	 */
	public Visualizer1D(FrameSource source, String title, boolean log) {
		this.title = title;
		this.source = source;
		this.log = log;
		fs = source.getFrameSize();
		buf = new double [fs];
		setPreferredSize(DEFAULT_SIZE);
	}
	
	/** 
	 * Return the outgoing frame size
	 */
	public int getFrameSize() {
		return fs;
	}

	/**
	 * Read the next frame, save it to buffer and repaint the plot
	 */
	public boolean read(double[] buf) throws IOException {
		if (!source.read(buf))
			return false;
		
		// copy the data
		
		System.arraycopy(buf, 0, this.buf, 0, fs);
		
		if (log)
			for (int i = 0; i < fs; ++i)
				this.buf[i] = Math.log(1E-6 + this.buf[i]);
		
		// repaint the scene
		repaint();
		
		return true;
	}
	
	private double max = Double.MAX_VALUE;
	private double min = Double.MIN_VALUE;
	
	/** min/max value BEFORE taking logarithm */
	public void setMinMax(double min, double max) {
		this.max = (log ? Math.log(1E-6 + max) : max);
		this.min = (log ? Math.log(1E-6 + min) : min);
	}
	
	public double getMax() { return max; }
	public double getMin() { return min; }
	
	public static int DEFAULT_BOXWIDTH = 3;
	
	/** 
	 * Visualization function
	 */
	public void paint(Graphics g) {
		// double buffering
		int w = getWidth();
		int h = getHeight();
		Image img = createImage(w, h);
		Graphics bg = img.getGraphics();
	
		// some background
		bg.setColor(Color.WHITE);
		bg.fillRect(0, 0, w, h);
		bg.setColor(Color.BLACK);
		
		// compute the scaling factor, mind clipping
		for (int i = 0; i < fs; ++i) {
			buf[i] = (buf[i] - min) / (max - min);
			if (buf[i] < 0.) buf[i] = 0.;
			if (buf[i] > 1.) buf[i] = 1.;
		}
		
		// box width, if required
		int bw = w / fs;
		
		if (min < 0. && max <= 0.) {
			// all values below zero
			for (int i = 0; i < fs; ++i) {
				if (w < fs * DEFAULT_BOXWIDTH)
					bg.drawLine(i, h, i, h - (int) (buf[i]*h));
				else
					bg.drawRect(i*bw, h - (int) (buf[i]*h), bw, (int) (buf[i]*h));
			}
		} else if (min < 0. && max > 0.) {
			// values below and above zero: plot zero line
			int ybase = (int)(h * (0. - min) / (max - min));
			bg.drawLine(0, ybase, w, ybase);
			
			// draw boxes/sticks
			for (int i = 0; i < fs; ++i) {
				if (w < fs * DEFAULT_BOXWIDTH)
					bg.drawLine(i, ybase, i, h - (int) (buf[i]*h));
				else
					bg.drawRect(i*bw, Math.min(ybase, h - (int) (buf[i]*h)), bw, Math.abs(ybase - h + (int) (buf[i]*h)));
			}
			
		} else if (min >= 0. && max >= 0.) {
			// all values above zero
			for (int i = 0; i < fs; ++i) {
				if (w < fs * DEFAULT_BOXWIDTH)
					bg.drawLine(i, h, i, h - (int) (buf[i]*h));
				else
					bg.drawRect(i*bw, h - (int) (buf[i]*h), bw, (int) (buf[i]*h));
			}
		}
	
		// plot image
		g.drawImage(img, 0, 0, this);
		g.setColor(Color.RED);
		g.drawString(title, 5, 10);
	}
}
