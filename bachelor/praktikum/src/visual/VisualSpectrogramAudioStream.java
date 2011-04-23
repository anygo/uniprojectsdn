package visual;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.image.BufferedImage;
import java.io.IOException;
import framed.FrameSource;

/**
 * VisualSpectrogramAudioStream implements a visualization of the spectrogram 
 * processing a given input stream
 * 
 * @author sidoneum, sifeluga
 * 
 */

public class VisualSpectrogramAudioStream extends VisualSpectrogram implements FrameSource, MouseListener, MouseMotionListener, MouseWheelListener {
	
	private static final long serialVersionUID = 1L;
	
	/**
	 * a reference to the input stream that provides our data
	 */
	protected FrameSource source = null;
	
	/**
	 * two dimensional ring buffered storage of double values
	 * representing an array where each elements holds 2^n + 1 FFT values
	 */
	protected double[][] ringbuf = null;
	
	/**
	 * current writing position inside the ring buffer
	 */
	protected int bufferIndex = 0;
	
	/**
	 * sample rate of the input stream
	 */
	protected int sampleRate = 0;
	
	/**
	 * current horizontal position of the mouse cursor with respect to the component
	 */
	protected int mouseX;
	
	/**
	 * current vertical position of the mouse cursor with respect to the component
	 */
	protected int mouseY;
	
	/**
	 * indicates if the mouse cursor is inside the component
	 */
	protected boolean mouseInside;
	
	/**
	 * shift length in milliseconds used to 'window' the input stream
	 */
	protected int shift;

	
///////////////////////////////////////////////////////////////////////////////
///								constructors								///
///////////////////////////////////////////////////////////////////////////////

	/**
	 * @param frameSource is the input stream to read from
	 * @param sampleRate is the sampling rate of the FrameSource
	 * @param shift length in milliseconds	
	 * @param bufferSize initial size of the ring buffer
	 */
	public VisualSpectrogramAudioStream(FrameSource frameSource, int sampleRate, int shift, int bufferSize) {
		
		this.colored = false;
		this.log = true;
		this.shift = shift;
		this.source = frameSource;
		this.maxFrequency = sampleRate/2;
		this.frameSize = frameSource.getFrameSize();
		this.ringbuf = new double[bufferSize][frameSize];
		setContrast(10.);
		
		setPreferredSize(new Dimension(512, 256));
		image = new BufferedImage(512, 256, BufferedImage.TYPE_INT_ARGB);
		addMouseListener(this);
		addMouseMotionListener(this);
		addMouseWheelListener(this);
	}


///////////////////////////////////////////////////////////////////////////////
///							set- and get-methods							///
///////////////////////////////////////////////////////////////////////////////
	
	
	/**
	 * sets the maximal time that can be displayed at once in the image
	 * @param ms time in milliseconds
	 * @see setBufferSize
	 */
	public void setTimeToShow(int ms) {
		
		setBufferSize(ms / shift);
	}
	
	/**
	 * sets and allocates the size of the ring buffer
	 * thereby bufferIndex will be reseted and old data will be overwritten
	 * @param bSize number of frames to store
	 */
	protected void setBufferSize(int bSize) {
		
		if (bSize <= 0) {
			bSize = 256;
			System.err.println("setBufferSize: bSize must be greater than 0, set to default value (256)");
		}
		
		bufferIndex = 0;
		ringbuf = new double[bSize][frameSize];	
	}


///////////////////////////////////////////////////////////////////////////////
///			methods for computing and visualizing spectrogram data			///
///////////////////////////////////////////////////////////////////////////////
	
	@Override
	public boolean read(double[] buf) throws IOException {
		
		// read from source
		if (!source.read(ringbuf[bufferIndex]))
			return false;
		
		// save to outgoing buffer
		System.arraycopy(ringbuf[bufferIndex], 0, buf, 0, buf.length);
		
		//bufferedPaint();
		repaint();
			
		// increment index
		bufferIndex = (bufferIndex + 1) % ringbuf.length;
		
		return true;
	}
	
	@Override
	public void paint(Graphics g) {
		
		bufferedPaint();
		
		g.drawImage(image, 0, 0, this);
		paintOverlay(g);
	}
	
	
	/**
	 * provides additional information of the processed data stream
	 * and paints them in the component, i.e. not in the image
	 * @param g graphics object
	 */
	protected void paintOverlay(Graphics g) {
		
		g.setColor(textColor);
		g.drawString("Spectrogram", image.getWidth() - 85, 20);
		
		String str = (shift * ringbuf.length) + " ms";
		g.drawString(str, image.getWidth() - str.length()*8 - 10, image.getHeight() - 10);
		
		if (mouseInside) {
			// print Scale
			g.setColor(scaleColor);
			g.drawLine(10, 0, 10, image.getHeight());
			g.drawLine(8, image.getHeight()-1, 12, image.getHeight()-1);
			
			
			double ratio = (double)image.getHeight()/(double)maxFrequency;
			for (int i = 0; i < maxFrequency; i += 1000) {
				int y = image.getHeight() - (int)(ratio * i) - 1;
				g.drawLine(8, y, 12, y);
			}
			
			// Mouse Events
		
			g.drawLine(image.getWidth() - 1, mouseY - 1, 8, mouseY - 1);
			ratio = 1. - (double)mouseY/(double)image.getHeight();
			g.drawString((int)(ratio * maxFrequency) + " Hz", 13, mouseY - 1);
		}
	}

	
	/**
	 * transfers the data from the ring buffer into the image
	 * and also controls color, brightness and contrast of the image
	 */
	public void bufferedPaint() {
		if (image.getHeight() != getHeight() || image.getWidth() != getWidth()) {
			setImageDimension(getWidth(), getHeight());
		}
		for (int x = 0; x < image.getWidth(); ++x) {
			
			// start at oldest value
			double ratio = (double)ringbuf.length / (double)image.getWidth();
			int bufPos = (int)((double)x * ratio + (double)bufferIndex) % ringbuf.length;
		
			for (int y = 0; y < image.getHeight(); ++y) {
				// no interpolation, just indexing
				
				int ip = (int) (y * frameSize / (double) image.getHeight());
				
				double bufferValue = ringbuf[bufPos][ip];
				if(getLog()) {
					bufferValue = Math.log(1E-6 + bufferValue);
				}
				double value = (bufferValue - min) / (max - min);
				
				if (value > 1.) {
					value = 1.;
				}
				
				// scale to gray or color space
				int rgb;
				if (colored) {		
					rgb = Color.HSBtoRGB((float)(1.-value), (float)value, brightness);		
				} 
				else {
					rgb = Color.HSBtoRGB((float)(1.-value), 0.f, 1.f-(float)value*brightness);	
				}
				image.setRGB(x, image.getHeight() - y - 1, rgb);
			} 
		}
	}
	
	
	@Override
	public int getFrameSize() {
		
		return frameSize;
	}

	
///////////////////////////////////////////////////////////////////////////////
///					implementation of internal mouse listeners				///
///		provides a default functionality to control VisualSynchronizer		///
///////////////////////////////////////////////////////////////////////////////
	
	@Override
	public void mouseClicked(MouseEvent e) {
		int button = e.getButton(); 
		if (button == 1) {
			setTimeToShow(5000);
		}
		else if (button == 3) {
			//setTimeToShow(10000);
			setLog(!getLog());
			
		}
		else if (button == 2) {
			setColored(!getColored());
		}
		repaint();
	}

	@Override
	public void mouseEntered(MouseEvent e) {
		
		mouseInside = true;
		repaint();
	}

	@Override
	public void mouseExited(MouseEvent e) {
		
		mouseInside = false;	
		repaint();
	}

	@Override
	public void mousePressed(MouseEvent e) {}

	@Override
	public void mouseReleased(MouseEvent e){}

	@Override
	public void mouseDragged(MouseEvent e) {}

	@Override
	public void mouseMoved(MouseEvent e) {
		
		mouseX = e.getX();
		mouseY = e.getY();
		repaint();
	}

	@Override
	public void mouseWheelMoved(MouseWheelEvent e) {
		
		int rotated =  e.getWheelRotation();
		if (rotated != 0) {
			setContrast(getContrast() + (double)rotated);
			repaint();
		}
	}
	
}
